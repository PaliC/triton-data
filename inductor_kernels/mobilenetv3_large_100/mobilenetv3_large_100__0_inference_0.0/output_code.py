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
# Topologically Sorted Source Nodes: [x_146], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_146 => convolution_63
# Graph fragment:
#   %convolution_63 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/lt/cltiu3szd73oc7uccpazylbbmynvcroh24nhc4luunkmkbitxhos.py
# Topologically Sorted Source Nodes: [x_146], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_146 => convolution_63
# Graph fragment:
#   %convolution_63 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[64, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
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


# kernel path: /tmp/torchinductor_sahanp/37/c37xhc36c752pxgtw63bjc4btxqnf5awkmg4b2floiq3bc7v3ipu.py
# Topologically Sorted Source Nodes: [x_147, x_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_147 => add_132, mul_168, mul_169, sub_46
#   x_148 => add_133, clamp_max_29, clamp_min_29, div_29, mul_170
# Graph fragment:
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_63, %unsqueeze_369), kwargs = {})
#   %mul_168 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_371), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_168, %unsqueeze_373), kwargs = {})
#   %add_132 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_169, %unsqueeze_375), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_132, 3), kwargs = {})
#   %clamp_min_29 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_133, 0), kwargs = {})
#   %clamp_max_29 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_29, 6), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_132, %clamp_max_29), kwargs = {})
#   %div_29 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_170, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 16
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fa/cfabv4asu4vuvglntz7t4i2t2rffo2ozgu2voz2mb7zumld3kupt.py
# Topologically Sorted Source Nodes: [x_150, x_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_150 => add_135, mul_172, mul_173, sub_47
#   x_151 => relu_19
# Graph fragment:
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_64, %unsqueeze_377), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %unsqueeze_379), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %unsqueeze_381), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %unsqueeze_383), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_135,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 16
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


# kernel path: /tmp/torchinductor_sahanp/ft/cftc5nyk7ozmdlv7qaqvqbhwwqmikjmuoxj65mzue2wockir33as.py
# Topologically Sorted Source Nodes: [x_153, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_153 => add_137, mul_175, mul_176, sub_48
#   x_154 => add_138
# Graph fragment:
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_65, %unsqueeze_385), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_387), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %unsqueeze_389), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %unsqueeze_391), kwargs = {})
#   %add_138 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_137, %div_29), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 16
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ol/colm4t7uxadjjj3bbtvx26t35yqafsoqtipebh6mitk2v6yihogw.py
# Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_156 => add_140, mul_178, mul_179, sub_49
#   x_157 => relu_20
# Graph fragment:
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_66, %unsqueeze_393), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_178, %unsqueeze_397), kwargs = {})
#   %add_140 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_179, %unsqueeze_399), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_140,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/57/c57mbimf66bwd55pehtsi4j57yuhg4kkzl5umcs5luvprewqjm63.py
# Topologically Sorted Source Nodes: [x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_159 => add_142, mul_181, mul_182, sub_50
#   x_160 => relu_21
# Graph fragment:
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_67, %unsqueeze_401), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_403), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_181, %unsqueeze_405), kwargs = {})
#   %add_142 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_182, %unsqueeze_407), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_142,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_sahanp/jq/cjq5kczgeco7qbpivqs7efiboofzol5cj2or6jp3555zxglfthmg.py
# Topologically Sorted Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_162 => add_144, mul_184, mul_185, sub_51
# Graph fragment:
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_409), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_411), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %unsqueeze_413), kwargs = {})
#   %add_144 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_415), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 24
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rz/crzgoevttk7vcvz3y6pjvkyzavjbgtqfqpsgptpw3vrqzuu7kf5t.py
# Topologically Sorted Source Nodes: [x_164, x_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_164 => add_146, mul_187, mul_188, sub_52
#   x_165 => relu_22
# Graph fragment:
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_69, %unsqueeze_417), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_419), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_187, %unsqueeze_421), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_188, %unsqueeze_423), kwargs = {})
#   %relu_22 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_146,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 72
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


# kernel path: /tmp/torchinductor_sahanp/b2/cb2irnppvtcaodb7gnhcm3vu52dxiqg4ilnv62wynvcpmjfmsaea.py
# Topologically Sorted Source Nodes: [x_170, x_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_170 => add_150, mul_193, mul_194, sub_54
#   x_171 => add_151
# Graph fragment:
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_433), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_435), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_193, %unsqueeze_437), kwargs = {})
#   %add_150 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_194, %unsqueeze_439), kwargs = {})
#   %add_151 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_150, %add_144), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 24
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/w4/cw4s7gvonl6pbhcqmim2azmsg7vg37irmbg3wqgfewof3penze2k.py
# Topologically Sorted Source Nodes: [x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_176 => add_155, mul_199, mul_200, sub_56
#   x_177 => relu_25
# Graph fragment:
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_73, %unsqueeze_449), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %unsqueeze_451), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %unsqueeze_453), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %unsqueeze_455), kwargs = {})
#   %relu_25 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_155,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 72
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


# kernel path: /tmp/torchinductor_sahanp/b3/cb3j5cvvgpu2vb2sz4235v7ndvtq3oux4a4defvse5qpqilysnz5.py
# Topologically Sorted Source Nodes: [x_se_32], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_32 => mean_9
# Graph fragment:
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_25, [2, 3], True), kwargs = {})
triton_red_fused_mean_11 = async_compile.triton('triton_red_fused_mean_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_11(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4032
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 72
    x1 = (xindex // 72)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (8064*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/m7/cm73kusmvz6o5zbn5lt5rdkjekwaxpb72pvsa3rn3qb5ms6sw6ml.py
# Topologically Sorted Source Nodes: [x_se_32], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_32 => mean_9
# Graph fragment:
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_25, [2, 3], True), kwargs = {})
triton_per_fused_mean_12 = async_compile.triton('triton_per_fused_mean_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_12(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 72
    x1 = (xindex // 72)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (504*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dm/cdmzwkxicwoccwubpwfdtn2emegmvvl5htzzqz5r5w645saawjfr.py
# Topologically Sorted Source Nodes: [x_se_32, x_se_33, x_se_34], Original ATen: [aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_se_32 => mean_9
#   x_se_33 => convolution_74
#   x_se_34 => relu_26
# Graph fragment:
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_25, [2, 3], True), kwargs = {})
#   %convolution_74 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_9, %arg56_1, %arg57_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_26 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_74,), kwargs = {})
triton_poi_fused_convolution_mean_relu_13 = async_compile.triton('triton_poi_fused_convolution_mean_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4y/c4ycc5hdp5tn7y6bn3xfre5l23vp6xoqbjg3v5gh47tf7wj45yqc.py
# Topologically Sorted Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35, hardsigmoid_8, x_178], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_8 => add_156, clamp_max_30, clamp_min_30, div_30
#   x_178 => mul_201
#   x_se_32 => mean_9
#   x_se_33 => convolution_74
#   x_se_34 => relu_26
#   x_se_35 => convolution_75
# Graph fragment:
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_25, [2, 3], True), kwargs = {})
#   %convolution_74 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_9, %arg56_1, %arg57_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_26 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_74,), kwargs = {})
#   %convolution_75 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_26, %arg58_1, %arg59_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_156 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_75, 3), kwargs = {})
#   %clamp_min_30 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_156, 0), kwargs = {})
#   %clamp_max_30 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_30, 6), kwargs = {})
#   %div_30 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_30, 6), kwargs = {})
#   %mul_201 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_25, %div_30), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_14 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_14(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 72
    x2 = (xindex // 56448)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (72*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pj/cpj724nayrhlldsllawnww5dmpgoqwpjjxjfjzgftsl4lsc7pjt6.py
# Topologically Sorted Source Nodes: [x_180], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_180 => add_158, mul_203, mul_204, sub_57
# Graph fragment:
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_76, %unsqueeze_457), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_459), kwargs = {})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_203, %unsqueeze_461), kwargs = {})
#   %add_158 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_204, %unsqueeze_463), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 40
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4q/c4q43rays76oelekncfeoptayolfel5cbbzapkj4mdackm2avvs7.py
# Topologically Sorted Source Nodes: [x_182, x_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_182 => add_160, mul_206, mul_207, sub_58
#   x_183 => relu_27
# Graph fragment:
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_465), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_467), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_206, %unsqueeze_469), kwargs = {})
#   %add_160 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_207, %unsqueeze_471), kwargs = {})
#   %relu_27 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_160,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
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


# kernel path: /tmp/torchinductor_sahanp/rg/crgf6ryhiswyfpst5tkgjzyzplcusjzphrh6ouhskjrxpk63pxh5.py
# Topologically Sorted Source Nodes: [x_se_36], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_36 => mean_10
# Graph fragment:
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_28, [2, 3], True), kwargs = {})
triton_red_fused_mean_17 = async_compile.triton('triton_red_fused_mean_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_17(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6720
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 120
    x1 = (xindex // 120)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (13440*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/o5/co5pwdhclsjkzxptm5evbmyfr6gw4l4pre4bnb22gzfhiqykea37.py
# Topologically Sorted Source Nodes: [x_se_36], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_36 => mean_10
# Graph fragment:
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_28, [2, 3], True), kwargs = {})
triton_per_fused_mean_18 = async_compile.triton('triton_per_fused_mean_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_18(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 120
    x1 = (xindex // 120)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (840*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/et/cetpkn5tyy6uyodnbwwxscpwyhacj5i7kybktgsuslw2uyhe2fmy.py
# Topologically Sorted Source Nodes: [x_se_36, x_se_37, x_se_38], Original ATen: [aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_se_36 => mean_10
#   x_se_37 => convolution_79
#   x_se_38 => relu_29
# Graph fragment:
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_28, [2, 3], True), kwargs = {})
#   %convolution_79 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_10, %arg75_1, %arg76_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_29 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_79,), kwargs = {})
triton_poi_fused_convolution_mean_relu_19 = async_compile.triton('triton_poi_fused_convolution_mean_relu_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ev/cevm4joo6kn4cu7xn5xehxy5cuvnfok73jn537bwuaq25opiyduz.py
# Topologically Sorted Source Nodes: [x_se_36, x_se_37, x_se_38, x_se_39, hardsigmoid_9, x_187], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_9 => add_163, clamp_max_31, clamp_min_31, div_31
#   x_187 => mul_211
#   x_se_36 => mean_10
#   x_se_37 => convolution_79
#   x_se_38 => relu_29
#   x_se_39 => convolution_80
# Graph fragment:
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_28, [2, 3], True), kwargs = {})
#   %convolution_79 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_10, %arg75_1, %arg76_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_29 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_79,), kwargs = {})
#   %convolution_80 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_29, %arg77_1, %arg78_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_163 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_80, 3), kwargs = {})
#   %clamp_min_31 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_163, 0), kwargs = {})
#   %clamp_max_31 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_31, 6), kwargs = {})
#   %div_31 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_31, 6), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_28, %div_31), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_20 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_20(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 120
    x2 = (xindex // 94080)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (120*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wv/cwvxy3j3iy5t6ujvccyger5k7n22pcq26oj7mbl27ptitywsfjhm.py
# Topologically Sorted Source Nodes: [x_189, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_189 => add_165, mul_213, mul_214, sub_60
#   x_190 => add_166
# Graph fragment:
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_81, %unsqueeze_481), kwargs = {})
#   %mul_213 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_483), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_213, %unsqueeze_485), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_214, %unsqueeze_487), kwargs = {})
#   %add_166 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_165, %add_158), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 40
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pa/cpaua7tjwprgqfdchep7j3dezsxxmddtrfjst6sxf4rcd5iahajl.py
# Topologically Sorted Source Nodes: [x_202, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_202 => add_176, mul_226, mul_227, sub_64
#   x_203 => add_177, clamp_max_33, clamp_min_33, div_33, mul_228
# Graph fragment:
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_513), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_515), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_226, %unsqueeze_517), kwargs = {})
#   %add_176 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_227, %unsqueeze_519), kwargs = {})
#   %add_177 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_176, 3), kwargs = {})
#   %clamp_min_33 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_177, 0), kwargs = {})
#   %clamp_max_33 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_33, 6), kwargs = {})
#   %mul_228 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_176, %clamp_max_33), kwargs = {})
#   %div_33 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_228, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zz/czznftkftnui54jrwrwe3j35pslog76mwzxutkf643eqyefusym6.py
# Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_205 => add_179, mul_230, mul_231, sub_65
#   x_206 => add_180, clamp_max_34, clamp_min_34, div_34, mul_232
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_88, %unsqueeze_521), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_230, %unsqueeze_525), kwargs = {})
#   %add_179 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_231, %unsqueeze_527), kwargs = {})
#   %add_180 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_179, 3), kwargs = {})
#   %clamp_min_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_180, 0), kwargs = {})
#   %clamp_max_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_34, 6), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_179, %clamp_max_34), kwargs = {})
#   %div_34 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_232, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ml/cmlhq5kosb5ggsfkesv56wg2uh72zm3euxjoshhmskvylrg5tiaq.py
# Topologically Sorted Source Nodes: [x_208], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_208 => add_182, mul_234, mul_235, sub_66
# Graph fragment:
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_89, %unsqueeze_529), kwargs = {})
#   %mul_234 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_234, %unsqueeze_533), kwargs = {})
#   %add_182 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_235, %unsqueeze_535), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ah/cahnj66zmzthf5z73bb5bjbt7423eukfghpevwg4vwlttheksivs.py
# Topologically Sorted Source Nodes: [x_210, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_210 => add_184, mul_237, mul_238, sub_67
#   x_211 => add_185, clamp_max_35, clamp_min_35, div_35, mul_239
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_90, %unsqueeze_537), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_237, %unsqueeze_541), kwargs = {})
#   %add_184 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_238, %unsqueeze_543), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_184, 3), kwargs = {})
#   %clamp_min_35 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_185, 0), kwargs = {})
#   %clamp_max_35 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_35, 6), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_184, %clamp_max_35), kwargs = {})
#   %div_35 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_239, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 313600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ze/czej477guajoewkzwnubfwz2q7qwr3exxc6avsbufvkjmyjykfy2.py
# Topologically Sorted Source Nodes: [x_216, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_216 => add_190, mul_245, mul_246, sub_69
#   x_217 => add_191
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_92, %unsqueeze_553), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_245, %unsqueeze_557), kwargs = {})
#   %add_190 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_246, %unsqueeze_559), kwargs = {})
#   %add_191 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_190, %add_182), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ev/cev3zqhr6dxd65i2a2fv4hwo7gysgldat5zwa2jjd6atibae76wr.py
# Topologically Sorted Source Nodes: [x_219, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_219 => add_193, mul_248, mul_249, sub_70
#   x_220 => add_194, clamp_max_37, clamp_min_37, div_37, mul_250
# Graph fragment:
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_93, %unsqueeze_561), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_563), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_248, %unsqueeze_565), kwargs = {})
#   %add_193 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_249, %unsqueeze_567), kwargs = {})
#   %add_194 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_193, 3), kwargs = {})
#   %clamp_min_37 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_194, 0), kwargs = {})
#   %clamp_max_37 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_37, 6), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_193, %clamp_max_37), kwargs = {})
#   %div_37 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_250, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 184
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/m2/cm2xtpxfobdnk6mmk7qexu5e5tsv5kemgyumaqnqdvnaddnhwo46.py
# Topologically Sorted Source Nodes: [x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_237 => add_211, mul_270, mul_271, sub_76
#   x_238 => add_212, clamp_max_41, clamp_min_41, div_41, mul_272
# Graph fragment:
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_99, %unsqueeze_609), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %unsqueeze_611), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_270, %unsqueeze_613), kwargs = {})
#   %add_211 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_271, %unsqueeze_615), kwargs = {})
#   %add_212 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_211, 3), kwargs = {})
#   %clamp_min_41 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_212, 0), kwargs = {})
#   %clamp_max_41 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_41, 6), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_211, %clamp_max_41), kwargs = {})
#   %div_41 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_272, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/di/cdigo2dx6wmdwby7tupz77zbbiueuz5v34uob67ibxmjke5wl3q2.py
# Topologically Sorted Source Nodes: [x_240], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_240 => add_214, mul_274, mul_275, sub_77
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_617), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_274, %unsqueeze_621), kwargs = {})
#   %add_214 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_275, %unsqueeze_623), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ax/caxqojbelbc3hyiyofevhtgdfaajjkgqocxmlkoppoe5ouxx3hn3.py
# Topologically Sorted Source Nodes: [x_241, x_se_44], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_241 => add_215, clamp_max_42, clamp_min_42, div_42, mul_276
#   x_se_44 => mean_12
# Graph fragment:
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_214, 3), kwargs = {})
#   %clamp_min_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_215, 0), kwargs = {})
#   %clamp_max_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_42, 6), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_214, %clamp_max_42), kwargs = {})
#   %div_42 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_276, 6), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_42, [2, 3], True), kwargs = {})
triton_red_fused_hardswish_mean_30 = async_compile.triton('triton_red_fused_hardswish_mean_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_mean_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_hardswish_mean_30(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (47040*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 3.0
        tmp2 = tmp0 + tmp1
        tmp3 = 0.0
        tmp4 = triton_helpers.maximum(tmp2, tmp3)
        tmp5 = 6.0
        tmp6 = triton_helpers.minimum(tmp4, tmp5)
        tmp7 = tmp0 * tmp6
        tmp8 = 0.16666666666666666
        tmp9 = tmp7 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5v/c5v7eora5pxkg55hftcuxywqc7nrah6adqfwm4zxetim5ks7o3kz.py
# Topologically Sorted Source Nodes: [x_241, x_se_44], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_241 => add_215, clamp_max_42, clamp_min_42, div_42, mul_276
#   x_se_44 => mean_12
# Graph fragment:
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_214, 3), kwargs = {})
#   %clamp_min_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_215, 0), kwargs = {})
#   %clamp_max_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_42, 6), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_214, %clamp_max_42), kwargs = {})
#   %div_42 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_276, 6), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_42, [2, 3], True), kwargs = {})
triton_per_fused_hardswish_mean_31 = async_compile.triton('triton_per_fused_hardswish_mean_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_31(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 480
    x1 = (xindex // 480)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (960*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tq/ctq22zgywl2umcmeheyvuuo7thr52ryvwkzbr7gna2txs36ciib5.py
# Topologically Sorted Source Nodes: [x_241, x_se_44, x_se_45, x_se_46], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_241 => add_215, clamp_max_42, clamp_min_42, div_42, mul_276
#   x_se_44 => mean_12
#   x_se_45 => convolution_101
#   x_se_46 => relu_33
# Graph fragment:
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_214, 3), kwargs = {})
#   %clamp_min_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_215, 0), kwargs = {})
#   %clamp_max_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_42, 6), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_214, %clamp_max_42), kwargs = {})
#   %div_42 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_276, 6), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_42, [2, 3], True), kwargs = {})
#   %convolution_101 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_12, %arg173_1, %arg174_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_33 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_101,), kwargs = {})
triton_poi_fused_convolution_hardswish_mean_relu_32 = async_compile.triton('triton_poi_fused_convolution_hardswish_mean_relu_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardswish_mean_relu_32(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/an/cancoxpjsfkz4aclu6nnibhnx2cwkav33veydh2dkwis7vuhfuiq.py
# Topologically Sorted Source Nodes: [x_241, x_se_44, x_se_45, x_se_46, x_se_47, hardsigmoid_11, x_242], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_11 => add_216, clamp_max_43, clamp_min_43, div_43
#   x_241 => add_215, clamp_max_42, clamp_min_42, div_42, mul_276
#   x_242 => mul_277
#   x_se_44 => mean_12
#   x_se_45 => convolution_101
#   x_se_46 => relu_33
#   x_se_47 => convolution_102
# Graph fragment:
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_214, 3), kwargs = {})
#   %clamp_min_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_215, 0), kwargs = {})
#   %clamp_max_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_42, 6), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_214, %clamp_max_42), kwargs = {})
#   %div_42 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_276, 6), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_42, [2, 3], True), kwargs = {})
#   %convolution_101 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_12, %arg173_1, %arg174_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_33 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_101,), kwargs = {})
#   %convolution_102 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_33, %arg175_1, %arg176_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_216 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_102, 3), kwargs = {})
#   %clamp_min_43 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_216, 0), kwargs = {})
#   %clamp_max_43 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_43, 6), kwargs = {})
#   %div_43 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_43, 6), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_42, %div_43), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_33 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_33(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 480
    x2 = (xindex // 94080)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + (480*x2)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp1
    tmp14 = triton_helpers.maximum(tmp13, tmp3)
    tmp15 = triton_helpers.minimum(tmp14, tmp5)
    tmp16 = tmp15 * tmp8
    tmp17 = tmp9 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fb/cfbxhgnx4mzu6dwkhjsko4xygrojvtjsnczk6b3abtryhitovauf.py
# Topologically Sorted Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_244 => add_218, mul_279, mul_280, sub_78
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_103, %unsqueeze_625), kwargs = {})
#   %mul_279 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_279, %unsqueeze_629), kwargs = {})
#   %add_218 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_280, %unsqueeze_631), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/b7/cb7gq7jn6lc5lzikbhhdjxxglgrvy3tie2jjkgf7xoxncsm7arhl.py
# Topologically Sorted Source Nodes: [x_246, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_246 => add_220, mul_282, mul_283, sub_79
#   x_247 => add_221, clamp_max_44, clamp_min_44, div_44, mul_284
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_104, %unsqueeze_633), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_635), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_282, %unsqueeze_637), kwargs = {})
#   %add_220 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_283, %unsqueeze_639), kwargs = {})
#   %add_221 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_220, 3), kwargs = {})
#   %clamp_min_44 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_221, 0), kwargs = {})
#   %clamp_max_44 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_44, 6), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_220, %clamp_max_44), kwargs = {})
#   %div_44 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_284, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uy/cuypn3yeoplp6j6wqun7zzc3vt7h5rd2cle6xq2cps3abn5joghr.py
# Topologically Sorted Source Nodes: [x_249], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_249 => add_223, mul_286, mul_287, sub_80
# Graph fragment:
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_641), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_643), kwargs = {})
#   %mul_287 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_286, %unsqueeze_645), kwargs = {})
#   %add_223 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_287, %unsqueeze_647), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ht/chtvylcxkpvwaxghqglwimteb57is24nrcxzh4qzr3bbocuhr2y6.py
# Topologically Sorted Source Nodes: [x_250, x_se_48], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_250 => add_224, clamp_max_45, clamp_min_45, div_45, mul_288
#   x_se_48 => mean_13
# Graph fragment:
#   %add_224 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_223, 3), kwargs = {})
#   %clamp_min_45 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_224, 0), kwargs = {})
#   %clamp_max_45 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_45, 6), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_223, %clamp_max_45), kwargs = {})
#   %div_45 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_288, 6), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_45, [2, 3], True), kwargs = {})
triton_red_fused_hardswish_mean_37 = async_compile.triton('triton_red_fused_hardswish_mean_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_mean_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_hardswish_mean_37(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 3.0
        tmp2 = tmp0 + tmp1
        tmp3 = 0.0
        tmp4 = triton_helpers.maximum(tmp2, tmp3)
        tmp5 = 6.0
        tmp6 = triton_helpers.minimum(tmp4, tmp5)
        tmp7 = tmp0 * tmp6
        tmp8 = 0.16666666666666666
        tmp9 = tmp7 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ej/cejmkersy2yt3tqhdansekje76hva2yo46ikprlijizeqsy4ikxn.py
# Topologically Sorted Source Nodes: [x_250, x_se_48], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_250 => add_224, clamp_max_45, clamp_min_45, div_45, mul_288
#   x_se_48 => mean_13
# Graph fragment:
#   %add_224 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_223, 3), kwargs = {})
#   %clamp_min_45 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_224, 0), kwargs = {})
#   %clamp_max_45 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_45, 6), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_223, %clamp_max_45), kwargs = {})
#   %div_45 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_288, 6), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_45, [2, 3], True), kwargs = {})
triton_per_fused_hardswish_mean_38 = async_compile.triton('triton_per_fused_hardswish_mean_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_38(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (1344*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dl/cdlajlhh5ghzl2ubglh2a3haelaozuxmyneiqngsnk7k3obgugkr.py
# Topologically Sorted Source Nodes: [x_250, x_se_48, x_se_49, x_se_50], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_250 => add_224, clamp_max_45, clamp_min_45, div_45, mul_288
#   x_se_48 => mean_13
#   x_se_49 => convolution_106
#   x_se_50 => relu_34
# Graph fragment:
#   %add_224 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_223, 3), kwargs = {})
#   %clamp_min_45 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_224, 0), kwargs = {})
#   %clamp_max_45 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_45, 6), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_223, %clamp_max_45), kwargs = {})
#   %div_45 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_288, 6), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_45, [2, 3], True), kwargs = {})
#   %convolution_106 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_13, %arg192_1, %arg193_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_34 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_106,), kwargs = {})
triton_poi_fused_convolution_hardswish_mean_relu_39 = async_compile.triton('triton_poi_fused_convolution_hardswish_mean_relu_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardswish_mean_relu_39(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 168
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vl/cvlsgxsik6igpicqqbabyr4uvxj5p23tqenbvg4raap2ss5pjltx.py
# Topologically Sorted Source Nodes: [x_250, x_se_48, x_se_49, x_se_50, x_se_51, hardsigmoid_12, x_251], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_12 => add_225, clamp_max_46, clamp_min_46, div_46
#   x_250 => add_224, clamp_max_45, clamp_min_45, div_45, mul_288
#   x_251 => mul_289
#   x_se_48 => mean_13
#   x_se_49 => convolution_106
#   x_se_50 => relu_34
#   x_se_51 => convolution_107
# Graph fragment:
#   %add_224 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_223, 3), kwargs = {})
#   %clamp_min_45 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_224, 0), kwargs = {})
#   %clamp_max_45 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_45, 6), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_223, %clamp_max_45), kwargs = {})
#   %div_45 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_288, 6), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_45, [2, 3], True), kwargs = {})
#   %convolution_106 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_13, %arg192_1, %arg193_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_34 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_106,), kwargs = {})
#   %convolution_107 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_34, %arg194_1, %arg195_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_225 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_107, 3), kwargs = {})
#   %clamp_min_46 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_225, 0), kwargs = {})
#   %clamp_max_46 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_46, 6), kwargs = {})
#   %div_46 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_46, 6), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_45, %div_46), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_40 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_40(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 131712)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp1
    tmp14 = triton_helpers.maximum(tmp13, tmp3)
    tmp15 = triton_helpers.minimum(tmp14, tmp5)
    tmp16 = tmp15 * tmp8
    tmp17 = tmp9 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7m/c7ma5ttlu4z5r7r4plxhsq2pkp24cxuxdw33gvxoj7ssfqszqmj7.py
# Topologically Sorted Source Nodes: [x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_253 => add_227, mul_291, mul_292, sub_81
#   x_254 => add_228
# Graph fragment:
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_108, %unsqueeze_649), kwargs = {})
#   %mul_291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_651), kwargs = {})
#   %mul_292 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_291, %unsqueeze_653), kwargs = {})
#   %add_227 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_292, %unsqueeze_655), kwargs = {})
#   %add_228 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_227, %add_218), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_41', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ok/cokugejwgyobnj42xbxpu4rc3y46z2dszphyhq273reljfqcthtu.py
# Topologically Sorted Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_259 => add_233, mul_298, mul_299, sub_83
# Graph fragment:
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_110, %unsqueeze_665), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_667), kwargs = {})
#   %mul_299 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_298, %unsqueeze_669), kwargs = {})
#   %add_233 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_299, %unsqueeze_671), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_42', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 263424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tm/ctmsqfcmr2krfeb3jnam4o4qrdzlwdpr24aonfk24zutptuqvkpk.py
# Topologically Sorted Source Nodes: [x_260, x_se_52], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_260 => add_234, clamp_max_48, clamp_min_48, div_48, mul_300
#   x_se_52 => mean_14
# Graph fragment:
#   %add_234 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_233, 3), kwargs = {})
#   %clamp_min_48 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_234, 0), kwargs = {})
#   %clamp_max_48 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_48, 6), kwargs = {})
#   %mul_300 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_233, %clamp_max_48), kwargs = {})
#   %div_48 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_300, 6), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_48, [2, 3], True), kwargs = {})
triton_per_fused_hardswish_mean_43 = async_compile.triton('triton_per_fused_hardswish_mean_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_43(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (32928*x1)), rmask & xmask, other=0.0)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = 49.0
    tmp15 = tmp13 / tmp14
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/e7/ce74qvxq2eyujjqw7ckgugjhz3pd3it4pb2ybnjwjigpryfx7omf.py
# Topologically Sorted Source Nodes: [x_260, x_se_52, x_se_53, x_se_54, x_se_55, hardsigmoid_13, x_261], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_13 => add_235, clamp_max_49, clamp_min_49, div_49
#   x_260 => add_234, clamp_max_48, clamp_min_48, div_48, mul_300
#   x_261 => mul_301
#   x_se_52 => mean_14
#   x_se_53 => convolution_111
#   x_se_54 => relu_35
#   x_se_55 => convolution_112
# Graph fragment:
#   %add_234 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_233, 3), kwargs = {})
#   %clamp_min_48 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_234, 0), kwargs = {})
#   %clamp_max_48 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_48, 6), kwargs = {})
#   %mul_300 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_233, %clamp_max_48), kwargs = {})
#   %div_48 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_300, 6), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_48, [2, 3], True), kwargs = {})
#   %convolution_111 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_14, %arg211_1, %arg212_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_35 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_111,), kwargs = {})
#   %convolution_112 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_35, %arg213_1, %arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_235 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_112, 3), kwargs = {})
#   %clamp_min_49 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_235, 0), kwargs = {})
#   %clamp_max_49 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_49, 6), kwargs = {})
#   %div_49 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_49, 6), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_48, %div_49), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_44 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_44(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 263424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 32928)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp1
    tmp14 = triton_helpers.maximum(tmp13, tmp3)
    tmp15 = triton_helpers.minimum(tmp14, tmp5)
    tmp16 = tmp15 * tmp8
    tmp17 = tmp9 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ir/circb6ozncjd6wenzgawzcjr65rl6patfigelsuc3crd4mjhjkhk.py
# Topologically Sorted Source Nodes: [x_263], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_263 => add_237, mul_303, mul_304, sub_84
# Graph fragment:
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_113, %unsqueeze_673), kwargs = {})
#   %mul_303 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_675), kwargs = {})
#   %mul_304 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_303, %unsqueeze_677), kwargs = {})
#   %add_237 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_304, %unsqueeze_679), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fd/cfdgvrgrowddkntvxf7voqopnv2xyolpyez25smmzgwl4rbne7mm.py
# Topologically Sorted Source Nodes: [x_265, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_265 => add_239, mul_306, mul_307, sub_85
#   x_266 => add_240, clamp_max_50, clamp_min_50, div_50, mul_308
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_114, %unsqueeze_681), kwargs = {})
#   %mul_306 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_306, %unsqueeze_685), kwargs = {})
#   %add_239 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_307, %unsqueeze_687), kwargs = {})
#   %add_240 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_239, 3), kwargs = {})
#   %clamp_min_50 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_240, 0), kwargs = {})
#   %clamp_max_50 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_50, 6), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_239, %clamp_max_50), kwargs = {})
#   %div_50 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_308, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_46', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tf/ctfwijzzhxh5u4zzef2c3qbuuzwfaoodoocgsup6dthwqrd2odj7.py
# Topologically Sorted Source Nodes: [x_268], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_268 => add_242, mul_310, mul_311, sub_86
# Graph fragment:
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_689), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_86, %unsqueeze_691), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, %unsqueeze_693), kwargs = {})
#   %add_242 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_311, %unsqueeze_695), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g7/cg7k23wjs3u5rghdlfh2x4oserb4bttwuuafaalmbiskobjnvc6s.py
# Topologically Sorted Source Nodes: [x_269, x_se_56], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_269 => add_243, clamp_max_51, clamp_min_51, div_51, mul_312
#   x_se_56 => mean_15
# Graph fragment:
#   %add_243 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_242, 3), kwargs = {})
#   %clamp_min_51 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_243, 0), kwargs = {})
#   %clamp_max_51 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_51, 6), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_242, %clamp_max_51), kwargs = {})
#   %div_51 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_312, 6), kwargs = {})
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_51, [2, 3], True), kwargs = {})
triton_per_fused_hardswish_mean_48 = async_compile.triton('triton_per_fused_hardswish_mean_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_48(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 960
    x1 = (xindex // 960)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (47040*x1)), rmask & xmask, other=0.0)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = 49.0
    tmp15 = tmp13 / tmp14
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vt/cvtmxymbjjv7i537bh6mod36temn6qd75735ndrfawnkyitrbi42.py
# Topologically Sorted Source Nodes: [x_269, x_se_56, x_se_57, x_se_58], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_269 => add_243, clamp_max_51, clamp_min_51, div_51, mul_312
#   x_se_56 => mean_15
#   x_se_57 => convolution_116
#   x_se_58 => relu_36
# Graph fragment:
#   %add_243 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_242, 3), kwargs = {})
#   %clamp_min_51 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_243, 0), kwargs = {})
#   %clamp_max_51 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_51, 6), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_242, %clamp_max_51), kwargs = {})
#   %div_51 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_312, 6), kwargs = {})
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_51, [2, 3], True), kwargs = {})
#   %convolution_116 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_15, %arg230_1, %arg231_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_36 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_116,), kwargs = {})
triton_poi_fused_convolution_hardswish_mean_relu_49 = async_compile.triton('triton_poi_fused_convolution_hardswish_mean_relu_49', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardswish_mean_relu_49(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ih/cih3i42kogn5spty4konjhttomebm3kvvd4nvsehlhvgwqurlkur.py
# Topologically Sorted Source Nodes: [x_269, x_se_56, x_se_57, x_se_58, x_se_59, hardsigmoid_14, x_270], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_14 => add_244, clamp_max_52, clamp_min_52, div_52
#   x_269 => add_243, clamp_max_51, clamp_min_51, div_51, mul_312
#   x_270 => mul_313
#   x_se_56 => mean_15
#   x_se_57 => convolution_116
#   x_se_58 => relu_36
#   x_se_59 => convolution_117
# Graph fragment:
#   %add_243 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_242, 3), kwargs = {})
#   %clamp_min_51 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_243, 0), kwargs = {})
#   %clamp_max_51 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_51, 6), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_242, %clamp_max_51), kwargs = {})
#   %div_51 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_312, 6), kwargs = {})
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_51, [2, 3], True), kwargs = {})
#   %convolution_116 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_15, %arg230_1, %arg231_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_36 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_116,), kwargs = {})
#   %convolution_117 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_36, %arg232_1, %arg233_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_244 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_117, 3), kwargs = {})
#   %clamp_min_52 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_244, 0), kwargs = {})
#   %clamp_max_52 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_52, 6), kwargs = {})
#   %div_52 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_52, 6), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_51, %div_52), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_50 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_50(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 960
    x2 = (xindex // 47040)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + (960*x2)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 + tmp1
    tmp14 = triton_helpers.maximum(tmp13, tmp3)
    tmp15 = triton_helpers.minimum(tmp14, tmp5)
    tmp16 = tmp15 * tmp8
    tmp17 = tmp9 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/py/cpy7ky4njv66dmr67syoestceboblve766w2goki63ok3347i327.py
# Topologically Sorted Source Nodes: [x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_272 => add_246, mul_315, mul_316, sub_87
#   x_273 => add_247
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_118, %unsqueeze_697), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_315, %unsqueeze_701), kwargs = {})
#   %add_246 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_316, %unsqueeze_703), kwargs = {})
#   %add_247 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_246, %add_237), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), xmask)
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uh/cuhw5r7kvzflsq3miuimfenui3cdigea2qbonsj7by574ohcmif7.py
# Topologically Sorted Source Nodes: [x_286, x_287, x_288, x_289], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_286 => add_260, clamp_max_56, clamp_min_56, div_56, mul_332
#   x_287 => mean_17
#   x_288 => convolution_125
#   x_289 => add_261, clamp_max_57, clamp_min_57, div_57, mul_333
# Graph fragment:
#   %add_260 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_259, 3), kwargs = {})
#   %clamp_min_56 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_260, 0), kwargs = {})
#   %clamp_max_56 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_56, 6), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_259, %clamp_max_56), kwargs = {})
#   %div_56 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_332, 6), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_56, [-1, -2], True), kwargs = {})
#   %convolution_125 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_17, %arg263_1, %arg264_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_261 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_125, 3), kwargs = {})
#   %clamp_min_57 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_261, 0), kwargs = {})
#   %clamp_max_57 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_57, 6), kwargs = {})
#   %mul_333 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_125, %clamp_max_57), kwargs = {})
#   %div_57 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_333, 6), kwargs = {})
triton_poi_fused_convolution_hardswish_mean_52 = async_compile.triton('triton_poi_fused_convolution_hardswish_mean_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardswish_mean_52(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = 0.16666666666666666
    tmp11 = tmp9 * tmp10
    tl.store(in_out_ptr0 + (x2), tmp11, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (16, ), (1, ))
    assert_size_stride(arg9_1, (16, ), (1, ))
    assert_size_stride(arg10_1, (16, ), (1, ))
    assert_size_stride(arg11_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg12_1, (16, ), (1, ))
    assert_size_stride(arg13_1, (16, ), (1, ))
    assert_size_stride(arg14_1, (16, ), (1, ))
    assert_size_stride(arg15_1, (16, ), (1, ))
    assert_size_stride(arg16_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg27_1, (24, ), (1, ))
    assert_size_stride(arg28_1, (24, ), (1, ))
    assert_size_stride(arg29_1, (24, ), (1, ))
    assert_size_stride(arg30_1, (24, ), (1, ))
    assert_size_stride(arg31_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg32_1, (72, ), (1, ))
    assert_size_stride(arg33_1, (72, ), (1, ))
    assert_size_stride(arg34_1, (72, ), (1, ))
    assert_size_stride(arg35_1, (72, ), (1, ))
    assert_size_stride(arg36_1, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg37_1, (72, ), (1, ))
    assert_size_stride(arg38_1, (72, ), (1, ))
    assert_size_stride(arg39_1, (72, ), (1, ))
    assert_size_stride(arg40_1, (72, ), (1, ))
    assert_size_stride(arg41_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg42_1, (24, ), (1, ))
    assert_size_stride(arg43_1, (24, ), (1, ))
    assert_size_stride(arg44_1, (24, ), (1, ))
    assert_size_stride(arg45_1, (24, ), (1, ))
    assert_size_stride(arg46_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg47_1, (72, ), (1, ))
    assert_size_stride(arg48_1, (72, ), (1, ))
    assert_size_stride(arg49_1, (72, ), (1, ))
    assert_size_stride(arg50_1, (72, ), (1, ))
    assert_size_stride(arg51_1, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg52_1, (72, ), (1, ))
    assert_size_stride(arg53_1, (72, ), (1, ))
    assert_size_stride(arg54_1, (72, ), (1, ))
    assert_size_stride(arg55_1, (72, ), (1, ))
    assert_size_stride(arg56_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg57_1, (24, ), (1, ))
    assert_size_stride(arg58_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg59_1, (72, ), (1, ))
    assert_size_stride(arg60_1, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg61_1, (40, ), (1, ))
    assert_size_stride(arg62_1, (40, ), (1, ))
    assert_size_stride(arg63_1, (40, ), (1, ))
    assert_size_stride(arg64_1, (40, ), (1, ))
    assert_size_stride(arg65_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg66_1, (120, ), (1, ))
    assert_size_stride(arg67_1, (120, ), (1, ))
    assert_size_stride(arg68_1, (120, ), (1, ))
    assert_size_stride(arg69_1, (120, ), (1, ))
    assert_size_stride(arg70_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg71_1, (120, ), (1, ))
    assert_size_stride(arg72_1, (120, ), (1, ))
    assert_size_stride(arg73_1, (120, ), (1, ))
    assert_size_stride(arg74_1, (120, ), (1, ))
    assert_size_stride(arg75_1, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg76_1, (32, ), (1, ))
    assert_size_stride(arg77_1, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg78_1, (120, ), (1, ))
    assert_size_stride(arg79_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg80_1, (40, ), (1, ))
    assert_size_stride(arg81_1, (40, ), (1, ))
    assert_size_stride(arg82_1, (40, ), (1, ))
    assert_size_stride(arg83_1, (40, ), (1, ))
    assert_size_stride(arg84_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg85_1, (120, ), (1, ))
    assert_size_stride(arg86_1, (120, ), (1, ))
    assert_size_stride(arg87_1, (120, ), (1, ))
    assert_size_stride(arg88_1, (120, ), (1, ))
    assert_size_stride(arg89_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg90_1, (120, ), (1, ))
    assert_size_stride(arg91_1, (120, ), (1, ))
    assert_size_stride(arg92_1, (120, ), (1, ))
    assert_size_stride(arg93_1, (120, ), (1, ))
    assert_size_stride(arg94_1, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg95_1, (32, ), (1, ))
    assert_size_stride(arg96_1, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg97_1, (120, ), (1, ))
    assert_size_stride(arg98_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg99_1, (40, ), (1, ))
    assert_size_stride(arg100_1, (40, ), (1, ))
    assert_size_stride(arg101_1, (40, ), (1, ))
    assert_size_stride(arg102_1, (40, ), (1, ))
    assert_size_stride(arg103_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg104_1, (240, ), (1, ))
    assert_size_stride(arg105_1, (240, ), (1, ))
    assert_size_stride(arg106_1, (240, ), (1, ))
    assert_size_stride(arg107_1, (240, ), (1, ))
    assert_size_stride(arg108_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg109_1, (240, ), (1, ))
    assert_size_stride(arg110_1, (240, ), (1, ))
    assert_size_stride(arg111_1, (240, ), (1, ))
    assert_size_stride(arg112_1, (240, ), (1, ))
    assert_size_stride(arg113_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg114_1, (80, ), (1, ))
    assert_size_stride(arg115_1, (80, ), (1, ))
    assert_size_stride(arg116_1, (80, ), (1, ))
    assert_size_stride(arg117_1, (80, ), (1, ))
    assert_size_stride(arg118_1, (200, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg119_1, (200, ), (1, ))
    assert_size_stride(arg120_1, (200, ), (1, ))
    assert_size_stride(arg121_1, (200, ), (1, ))
    assert_size_stride(arg122_1, (200, ), (1, ))
    assert_size_stride(arg123_1, (200, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg124_1, (200, ), (1, ))
    assert_size_stride(arg125_1, (200, ), (1, ))
    assert_size_stride(arg126_1, (200, ), (1, ))
    assert_size_stride(arg127_1, (200, ), (1, ))
    assert_size_stride(arg128_1, (80, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg129_1, (80, ), (1, ))
    assert_size_stride(arg130_1, (80, ), (1, ))
    assert_size_stride(arg131_1, (80, ), (1, ))
    assert_size_stride(arg132_1, (80, ), (1, ))
    assert_size_stride(arg133_1, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg134_1, (184, ), (1, ))
    assert_size_stride(arg135_1, (184, ), (1, ))
    assert_size_stride(arg136_1, (184, ), (1, ))
    assert_size_stride(arg137_1, (184, ), (1, ))
    assert_size_stride(arg138_1, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg139_1, (184, ), (1, ))
    assert_size_stride(arg140_1, (184, ), (1, ))
    assert_size_stride(arg141_1, (184, ), (1, ))
    assert_size_stride(arg142_1, (184, ), (1, ))
    assert_size_stride(arg143_1, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg144_1, (80, ), (1, ))
    assert_size_stride(arg145_1, (80, ), (1, ))
    assert_size_stride(arg146_1, (80, ), (1, ))
    assert_size_stride(arg147_1, (80, ), (1, ))
    assert_size_stride(arg148_1, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg149_1, (184, ), (1, ))
    assert_size_stride(arg150_1, (184, ), (1, ))
    assert_size_stride(arg151_1, (184, ), (1, ))
    assert_size_stride(arg152_1, (184, ), (1, ))
    assert_size_stride(arg153_1, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg154_1, (184, ), (1, ))
    assert_size_stride(arg155_1, (184, ), (1, ))
    assert_size_stride(arg156_1, (184, ), (1, ))
    assert_size_stride(arg157_1, (184, ), (1, ))
    assert_size_stride(arg158_1, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg159_1, (80, ), (1, ))
    assert_size_stride(arg160_1, (80, ), (1, ))
    assert_size_stride(arg161_1, (80, ), (1, ))
    assert_size_stride(arg162_1, (80, ), (1, ))
    assert_size_stride(arg163_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg164_1, (480, ), (1, ))
    assert_size_stride(arg165_1, (480, ), (1, ))
    assert_size_stride(arg166_1, (480, ), (1, ))
    assert_size_stride(arg167_1, (480, ), (1, ))
    assert_size_stride(arg168_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg169_1, (480, ), (1, ))
    assert_size_stride(arg170_1, (480, ), (1, ))
    assert_size_stride(arg171_1, (480, ), (1, ))
    assert_size_stride(arg172_1, (480, ), (1, ))
    assert_size_stride(arg173_1, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg174_1, (120, ), (1, ))
    assert_size_stride(arg175_1, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg176_1, (480, ), (1, ))
    assert_size_stride(arg177_1, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg178_1, (112, ), (1, ))
    assert_size_stride(arg179_1, (112, ), (1, ))
    assert_size_stride(arg180_1, (112, ), (1, ))
    assert_size_stride(arg181_1, (112, ), (1, ))
    assert_size_stride(arg182_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg183_1, (672, ), (1, ))
    assert_size_stride(arg184_1, (672, ), (1, ))
    assert_size_stride(arg185_1, (672, ), (1, ))
    assert_size_stride(arg186_1, (672, ), (1, ))
    assert_size_stride(arg187_1, (672, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg188_1, (672, ), (1, ))
    assert_size_stride(arg189_1, (672, ), (1, ))
    assert_size_stride(arg190_1, (672, ), (1, ))
    assert_size_stride(arg191_1, (672, ), (1, ))
    assert_size_stride(arg192_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg193_1, (168, ), (1, ))
    assert_size_stride(arg194_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg195_1, (672, ), (1, ))
    assert_size_stride(arg196_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg197_1, (112, ), (1, ))
    assert_size_stride(arg198_1, (112, ), (1, ))
    assert_size_stride(arg199_1, (112, ), (1, ))
    assert_size_stride(arg200_1, (112, ), (1, ))
    assert_size_stride(arg201_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg202_1, (672, ), (1, ))
    assert_size_stride(arg203_1, (672, ), (1, ))
    assert_size_stride(arg204_1, (672, ), (1, ))
    assert_size_stride(arg205_1, (672, ), (1, ))
    assert_size_stride(arg206_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg207_1, (672, ), (1, ))
    assert_size_stride(arg208_1, (672, ), (1, ))
    assert_size_stride(arg209_1, (672, ), (1, ))
    assert_size_stride(arg210_1, (672, ), (1, ))
    assert_size_stride(arg211_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg212_1, (168, ), (1, ))
    assert_size_stride(arg213_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg214_1, (672, ), (1, ))
    assert_size_stride(arg215_1, (160, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg216_1, (160, ), (1, ))
    assert_size_stride(arg217_1, (160, ), (1, ))
    assert_size_stride(arg218_1, (160, ), (1, ))
    assert_size_stride(arg219_1, (160, ), (1, ))
    assert_size_stride(arg220_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg221_1, (960, ), (1, ))
    assert_size_stride(arg222_1, (960, ), (1, ))
    assert_size_stride(arg223_1, (960, ), (1, ))
    assert_size_stride(arg224_1, (960, ), (1, ))
    assert_size_stride(arg225_1, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg226_1, (960, ), (1, ))
    assert_size_stride(arg227_1, (960, ), (1, ))
    assert_size_stride(arg228_1, (960, ), (1, ))
    assert_size_stride(arg229_1, (960, ), (1, ))
    assert_size_stride(arg230_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg231_1, (240, ), (1, ))
    assert_size_stride(arg232_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg233_1, (960, ), (1, ))
    assert_size_stride(arg234_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg235_1, (160, ), (1, ))
    assert_size_stride(arg236_1, (160, ), (1, ))
    assert_size_stride(arg237_1, (160, ), (1, ))
    assert_size_stride(arg238_1, (160, ), (1, ))
    assert_size_stride(arg239_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg240_1, (960, ), (1, ))
    assert_size_stride(arg241_1, (960, ), (1, ))
    assert_size_stride(arg242_1, (960, ), (1, ))
    assert_size_stride(arg243_1, (960, ), (1, ))
    assert_size_stride(arg244_1, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg245_1, (960, ), (1, ))
    assert_size_stride(arg246_1, (960, ), (1, ))
    assert_size_stride(arg247_1, (960, ), (1, ))
    assert_size_stride(arg248_1, (960, ), (1, ))
    assert_size_stride(arg249_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg250_1, (240, ), (1, ))
    assert_size_stride(arg251_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg252_1, (960, ), (1, ))
    assert_size_stride(arg253_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg254_1, (160, ), (1, ))
    assert_size_stride(arg255_1, (160, ), (1, ))
    assert_size_stride(arg256_1, (160, ), (1, ))
    assert_size_stride(arg257_1, (160, ), (1, ))
    assert_size_stride(arg258_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg259_1, (960, ), (1, ))
    assert_size_stride(arg260_1, (960, ), (1, ))
    assert_size_stride(arg261_1, (960, ), (1, ))
    assert_size_stride(arg262_1, (960, ), (1, ))
    assert_size_stride(arg263_1, (1280, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg264_1, (1280, ), (1, ))
    assert_size_stride(arg265_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg266_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((16, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 48, 9, grid=grid(48, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((8, 16, 112, 112), (200704, 1, 1792, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_147, x_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, buf4, 1605632, grid=grid(1605632), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf3
        # Topologically Sorted Source Nodes: [x_148, x_149], Original ATen: [aten.hardswish, aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf5, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del arg6_1
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_150, x_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [x_150, x_151, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del arg11_1
        del buf6
        buf8 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_153, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_4.run(buf8, buf7, arg12_1, arg13_1, arg14_1, arg15_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del buf7
        # Topologically Sorted Source Nodes: [x_153, x_154, x_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 64, 112, 112), (802816, 1, 7168, 64))
        del arg16_1
        del buf8
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf10, arg17_1, arg18_1, arg19_1, arg20_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        # Topologically Sorted Source Nodes: [x_156, x_157, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg21_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf11, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg21_1
        del buf10
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf12, arg22_1, arg23_1, arg24_1, arg25_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        # Topologically Sorted Source Nodes: [x_159, x_160, x_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg26_1
        del buf12
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_7.run(buf14, arg27_1, arg28_1, arg29_1, arg30_1, 602112, grid=grid(602112), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 72, 56, 56), (225792, 1, 4032, 72))
        del arg31_1
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_164, x_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf16, arg32_1, arg33_1, arg34_1, arg35_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        # Topologically Sorted Source Nodes: [x_164, x_165, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf17 = extern_kernels.convolution(buf16, arg36_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf17, (8, 72, 56, 56), (225792, 1, 4032, 72))
        del arg36_1
        del buf16
        buf18 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [x_167, x_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf18, arg37_1, arg38_1, arg39_1, arg40_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        # Topologically Sorted Source Nodes: [x_167, x_168, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg41_1
        del buf18
        buf20 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_170, x_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_9.run(buf20, buf19, arg42_1, arg43_1, arg44_1, arg45_1, 602112, grid=grid(602112), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        del buf19
        # Topologically Sorted Source Nodes: [x_170, x_171, x_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 72, 56, 56), (225792, 1, 4032, 72))
        del arg46_1
        del buf20
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_173, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf22, arg47_1, arg48_1, arg49_1, arg50_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        # Topologically Sorted Source Nodes: [x_173, x_174, x_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg51_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf23, (8, 72, 28, 28), (56448, 1, 2016, 72))
        del arg51_1
        del buf22
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf24, arg52_1, arg53_1, arg54_1, arg55_1, 451584, grid=grid(451584), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        buf25 = empty_strided_cuda((8, 72, 1, 1, 7), (504, 1, 4032, 4032, 72), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_32], Original ATen: [aten.mean]
        triton_red_fused_mean_11.run(buf24, buf25, 4032, 112, grid=grid(4032), stream=stream0)
        buf27 = empty_strided_cuda((8, 72, 1, 1), (72, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_32], Original ATen: [aten.mean]
        triton_per_fused_mean_12.run(buf25, buf27, 576, 7, grid=grid(576), stream=stream0)
        del buf25
        # Topologically Sorted Source Nodes: [x_se_32, x_se_33], Original ATen: [aten.mean, aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 24, 1, 1), (24, 1, 1, 1))
        del arg56_1
        del buf27
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_se_32, x_se_33, x_se_34], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_13.run(buf29, arg57_1, 192, grid=grid(192), stream=stream0)
        del arg57_1
        # Topologically Sorted Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf30 = extern_kernels.convolution(buf29, arg58_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 72, 1, 1), (72, 1, 1, 1))
        del arg58_1
        del buf29
        buf31 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35, hardsigmoid_8, x_178], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_14.run(buf31, buf30, arg59_1, 451584, grid=grid(451584), stream=stream0)
        del arg59_1
        del buf30
        # Topologically Sorted Source Nodes: [x_se_32, x_se_33, x_se_34, x_se_35, hardsigmoid_8, x_178, x_179], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf32 = extern_kernels.convolution(buf31, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 40, 28, 28), (31360, 1, 1120, 40))
        del arg60_1
        del buf31
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_180], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf33, arg61_1, arg62_1, arg63_1, arg64_1, 250880, grid=grid(250880), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        del arg64_1
        # Topologically Sorted Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg65_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 120, 28, 28), (94080, 1, 3360, 120))
        del arg65_1
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_182, x_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf35, arg66_1, arg67_1, arg68_1, arg69_1, 752640, grid=grid(752640), stream=stream0)
        del arg66_1
        del arg67_1
        del arg68_1
        del arg69_1
        # Topologically Sorted Source Nodes: [x_182, x_183, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg70_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf36, (8, 120, 28, 28), (94080, 1, 3360, 120))
        del arg70_1
        del buf35
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_185, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf37, arg71_1, arg72_1, arg73_1, arg74_1, 752640, grid=grid(752640), stream=stream0)
        del arg71_1
        del arg72_1
        del arg73_1
        del arg74_1
        buf38 = empty_strided_cuda((8, 120, 1, 1, 7), (840, 1, 6720, 6720, 120), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_36], Original ATen: [aten.mean]
        triton_red_fused_mean_17.run(buf37, buf38, 6720, 112, grid=grid(6720), stream=stream0)
        buf40 = empty_strided_cuda((8, 120, 1, 1), (120, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_36], Original ATen: [aten.mean]
        triton_per_fused_mean_18.run(buf38, buf40, 960, 7, grid=grid(960), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_36, x_se_37], Original ATen: [aten.mean, aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg75_1
        del buf40
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_se_36, x_se_37, x_se_38], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_19.run(buf42, arg76_1, 256, grid=grid(256), stream=stream0)
        del arg76_1
        # Topologically Sorted Source Nodes: [x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf43 = extern_kernels.convolution(buf42, arg77_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg77_1
        del buf42
        buf44 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_se_36, x_se_37, x_se_38, x_se_39, hardsigmoid_9, x_187], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_20.run(buf44, buf43, arg78_1, 752640, grid=grid(752640), stream=stream0)
        del arg78_1
        # Topologically Sorted Source Nodes: [x_se_36, x_se_37, x_se_38, x_se_39, hardsigmoid_9, x_187, x_188], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf45 = extern_kernels.convolution(buf44, arg79_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 40, 28, 28), (31360, 1, 1120, 40))
        del arg79_1
        del buf44
        buf46 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_189, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf46, buf45, arg80_1, arg81_1, arg82_1, arg83_1, 250880, grid=grid(250880), stream=stream0)
        del arg80_1
        del arg81_1
        del arg82_1
        del arg83_1
        del buf45
        # Topologically Sorted Source Nodes: [x_191], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 120, 28, 28), (94080, 1, 3360, 120))
        del arg84_1
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_192, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf48, arg85_1, arg86_1, arg87_1, arg88_1, 752640, grid=grid(752640), stream=stream0)
        del arg85_1
        del arg86_1
        del arg87_1
        del arg88_1
        # Topologically Sorted Source Nodes: [x_192, x_193, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg89_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf49, (8, 120, 28, 28), (94080, 1, 3360, 120))
        del arg89_1
        del buf48
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [x_195, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf50, arg90_1, arg91_1, arg92_1, arg93_1, 752640, grid=grid(752640), stream=stream0)
        del arg90_1
        del arg91_1
        del arg92_1
        del arg93_1
        buf51 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_se_40], Original ATen: [aten.mean]
        triton_red_fused_mean_17.run(buf50, buf51, 6720, 112, grid=grid(6720), stream=stream0)
        buf53 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_se_40], Original ATen: [aten.mean]
        triton_per_fused_mean_18.run(buf51, buf53, 960, 7, grid=grid(960), stream=stream0)
        del buf51
        # Topologically Sorted Source Nodes: [x_se_40, x_se_41], Original ATen: [aten.mean, aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg94_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg94_1
        del buf53
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_se_40, x_se_41, x_se_42], Original ATen: [aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_19.run(buf55, arg95_1, 256, grid=grid(256), stream=stream0)
        del arg95_1
        # Topologically Sorted Source Nodes: [x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.mean, aten.convolution, aten.relu]
        buf56 = extern_kernels.convolution(buf55, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg96_1
        del buf55
        buf57 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_se_40, x_se_41, x_se_42, x_se_43, hardsigmoid_10, x_197], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_20.run(buf57, buf56, arg97_1, 752640, grid=grid(752640), stream=stream0)
        del arg97_1
        del buf56
        # Topologically Sorted Source Nodes: [x_se_40, x_se_41, x_se_42, x_se_43, hardsigmoid_10, x_197, x_198], Original ATen: [aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf58 = extern_kernels.convolution(buf57, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 40, 28, 28), (31360, 1, 1120, 40))
        del arg98_1
        buf59 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_199, x_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf59, buf58, arg99_1, arg100_1, arg101_1, arg102_1, 250880, grid=grid(250880), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        del arg99_1
        del buf58
        # Topologically Sorted Source Nodes: [x_199, x_200, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg103_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 240, 28, 28), (188160, 1, 6720, 240))
        del arg103_1
        del buf59
        buf61 = buf60; del buf60  # reuse
        buf62 = empty_strided_cuda((8, 240, 28, 28), (188160, 1, 6720, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_202, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf61, arg104_1, arg105_1, arg106_1, arg107_1, buf62, 1505280, grid=grid(1505280), stream=stream0)
        del arg104_1
        del arg105_1
        del arg106_1
        del arg107_1
        del buf61
        # Topologically Sorted Source Nodes: [x_203, x_204], Original ATen: [aten.hardswish, aten.convolution]
        buf63 = extern_kernels.convolution(buf62, arg108_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf63, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg108_1
        del buf62
        buf64 = buf63; del buf63  # reuse
        buf65 = empty_strided_cuda((8, 240, 14, 14), (47040, 1, 3360, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_23.run(buf64, arg109_1, arg110_1, arg111_1, arg112_1, buf65, 376320, grid=grid(376320), stream=stream0)
        del arg109_1
        del arg110_1
        del arg111_1
        del arg112_1
        del buf64
        # Topologically Sorted Source Nodes: [x_206, x_207], Original ATen: [aten.hardswish, aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg113_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg113_1
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_208], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf67, arg114_1, arg115_1, arg116_1, arg117_1, 125440, grid=grid(125440), stream=stream0)
        del arg114_1
        del arg115_1
        del arg116_1
        del arg117_1
        # Topologically Sorted Source Nodes: [x_209], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg118_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 200, 14, 14), (39200, 1, 2800, 200))
        del arg118_1
        buf69 = buf68; del buf68  # reuse
        buf70 = empty_strided_cuda((8, 200, 14, 14), (39200, 1, 2800, 200), torch.float32)
        # Topologically Sorted Source Nodes: [x_210, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_25.run(buf69, arg119_1, arg120_1, arg121_1, arg122_1, buf70, 313600, grid=grid(313600), stream=stream0)
        del arg119_1
        del arg120_1
        del arg121_1
        del arg122_1
        del buf69
        # Topologically Sorted Source Nodes: [x_211, x_212], Original ATen: [aten.hardswish, aten.convolution]
        buf71 = extern_kernels.convolution(buf70, arg123_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=200, bias=None)
        assert_size_stride(buf71, (8, 200, 14, 14), (39200, 1, 2800, 200))
        del arg123_1
        buf72 = buf71; del buf71  # reuse
        buf73 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_213, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_25.run(buf72, arg124_1, arg125_1, arg126_1, arg127_1, buf73, 313600, grid=grid(313600), stream=stream0)
        del arg124_1
        del arg125_1
        del arg126_1
        del arg127_1
        del buf72
        # Topologically Sorted Source Nodes: [x_214, x_215], Original ATen: [aten.hardswish, aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg128_1
        del buf73
        buf75 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_216, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf75, buf74, arg129_1, arg130_1, arg131_1, arg132_1, 125440, grid=grid(125440), stream=stream0)
        del arg129_1
        del arg130_1
        del arg131_1
        del arg132_1
        del buf74
        # Topologically Sorted Source Nodes: [x_218], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 184, 14, 14), (36064, 1, 2576, 184))
        del arg133_1
        buf77 = buf76; del buf76  # reuse
        buf78 = empty_strided_cuda((8, 184, 14, 14), (36064, 1, 2576, 184), torch.float32)
        # Topologically Sorted Source Nodes: [x_219, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_27.run(buf77, arg134_1, arg135_1, arg136_1, arg137_1, buf78, 288512, grid=grid(288512), stream=stream0)
        del arg134_1
        del arg135_1
        del arg136_1
        del arg137_1
        del buf77
        # Topologically Sorted Source Nodes: [x_220, x_221], Original ATen: [aten.hardswish, aten.convolution]
        buf79 = extern_kernels.convolution(buf78, arg138_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
        assert_size_stride(buf79, (8, 184, 14, 14), (36064, 1, 2576, 184))
        del arg138_1
        buf80 = buf79; del buf79  # reuse
        buf81 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_222, x_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_27.run(buf80, arg139_1, arg140_1, arg141_1, arg142_1, buf81, 288512, grid=grid(288512), stream=stream0)
        del arg139_1
        del arg140_1
        del arg141_1
        del arg142_1
        del buf80
        # Topologically Sorted Source Nodes: [x_223, x_224], Original ATen: [aten.hardswish, aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg143_1
        buf83 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_225, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf83, buf82, arg144_1, arg145_1, arg146_1, arg147_1, 125440, grid=grid(125440), stream=stream0)
        del arg144_1
        del arg145_1
        del arg146_1
        del arg147_1
        del buf82
        # Topologically Sorted Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 184, 14, 14), (36064, 1, 2576, 184))
        del arg148_1
        buf85 = buf84; del buf84  # reuse
        buf86 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_228, x_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_27.run(buf85, arg149_1, arg150_1, arg151_1, arg152_1, buf86, 288512, grid=grid(288512), stream=stream0)
        del arg149_1
        del arg150_1
        del arg151_1
        del arg152_1
        del buf85
        # Topologically Sorted Source Nodes: [x_229, x_230], Original ATen: [aten.hardswish, aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
        assert_size_stride(buf87, (8, 184, 14, 14), (36064, 1, 2576, 184))
        del arg153_1
        buf88 = buf87; del buf87  # reuse
        buf89 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_231, x_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_27.run(buf88, arg154_1, arg155_1, arg156_1, arg157_1, buf89, 288512, grid=grid(288512), stream=stream0)
        del arg154_1
        del arg155_1
        del arg156_1
        del arg157_1
        del buf88
        # Topologically Sorted Source Nodes: [x_232, x_233], Original ATen: [aten.hardswish, aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg158_1
        del buf89
        buf91 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf91, buf90, arg159_1, arg160_1, arg161_1, arg162_1, 125440, grid=grid(125440), stream=stream0)
        del arg159_1
        del arg160_1
        del arg161_1
        del arg162_1
        del buf90
        # Topologically Sorted Source Nodes: [x_234, x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 480, 14, 14), (94080, 1, 6720, 480))
        del arg163_1
        del buf91
        buf93 = buf92; del buf92  # reuse
        buf94 = reinterpret_tensor(buf57, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_28.run(buf93, arg164_1, arg165_1, arg166_1, arg167_1, buf94, 752640, grid=grid(752640), stream=stream0)
        del arg164_1
        del arg165_1
        del arg166_1
        del arg167_1
        del buf93
        # Topologically Sorted Source Nodes: [x_238, x_239], Original ATen: [aten.hardswish, aten.convolution]
        buf95 = extern_kernels.convolution(buf94, arg168_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf95, (8, 480, 14, 14), (94080, 1, 6720, 480))
        del arg168_1
        del buf94
        buf96 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [x_240], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_29.run(buf96, arg169_1, arg170_1, arg171_1, arg172_1, 752640, grid=grid(752640), stream=stream0)
        del arg169_1
        del arg170_1
        del arg171_1
        del arg172_1
        buf97 = empty_strided_cuda((8, 480, 1, 1, 2), (960, 1, 7680, 7680, 480), torch.float32)
        # Topologically Sorted Source Nodes: [x_241, x_se_44], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_30.run(buf96, buf97, 7680, 98, grid=grid(7680), stream=stream0)
        buf99 = empty_strided_cuda((8, 480, 1, 1), (480, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_241, x_se_44], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_31.run(buf97, buf99, 3840, 2, grid=grid(3840), stream=stream0)
        # Topologically Sorted Source Nodes: [x_241, x_se_44, x_se_45], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg173_1
        del buf99
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_se_44, x_se_45, x_se_46], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_32.run(buf101, arg174_1, 960, grid=grid(960), stream=stream0)
        del arg174_1
        # Topologically Sorted Source Nodes: [x_241, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        buf102 = extern_kernels.convolution(buf101, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg175_1
        del buf101
        buf103 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_se_44, x_se_45, x_se_46, x_se_47, hardsigmoid_11, x_242], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_33.run(buf103, buf102, arg176_1, 752640, grid=grid(752640), stream=stream0)
        del arg176_1
        del buf102
        # Topologically Sorted Source Nodes: [x_241, x_se_44, x_se_45, x_se_46, x_se_47, hardsigmoid_11, x_242, x_243], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf104 = extern_kernels.convolution(buf103, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg177_1
        del buf103
        buf105 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_34.run(buf105, arg178_1, arg179_1, arg180_1, arg181_1, 175616, grid=grid(175616), stream=stream0)
        del arg178_1
        del arg179_1
        del arg180_1
        del arg181_1
        # Topologically Sorted Source Nodes: [x_245], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg182_1
        buf107 = buf106; del buf106  # reuse
        buf108 = empty_strided_cuda((8, 672, 14, 14), (131712, 1, 9408, 672), torch.float32)
        # Topologically Sorted Source Nodes: [x_246, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_35.run(buf107, arg183_1, arg184_1, arg185_1, arg186_1, buf108, 1053696, grid=grid(1053696), stream=stream0)
        del arg183_1
        del arg184_1
        del arg185_1
        del arg186_1
        del buf107
        # Topologically Sorted Source Nodes: [x_247, x_248], Original ATen: [aten.hardswish, aten.convolution]
        buf109 = extern_kernels.convolution(buf108, arg187_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf109, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg187_1
        del buf108
        buf110 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_249], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf110, arg188_1, arg189_1, arg190_1, arg191_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg188_1
        del arg189_1
        del arg190_1
        del arg191_1
        buf111 = empty_strided_cuda((8, 672, 1, 1, 2), (1344, 1, 10752, 10752, 672), torch.float32)
        # Topologically Sorted Source Nodes: [x_250, x_se_48], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_37.run(buf110, buf111, 10752, 98, grid=grid(10752), stream=stream0)
        buf113 = empty_strided_cuda((8, 672, 1, 1), (672, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_250, x_se_48], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_38.run(buf111, buf113, 5376, 2, grid=grid(5376), stream=stream0)
        del buf111
        # Topologically Sorted Source Nodes: [x_250, x_se_48, x_se_49], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf114 = extern_kernels.convolution(buf113, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 168, 1, 1), (168, 1, 1, 1))
        del arg192_1
        del buf113
        buf115 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [x_250, x_se_48, x_se_49, x_se_50], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_39.run(buf115, arg193_1, 1344, grid=grid(1344), stream=stream0)
        del arg193_1
        # Topologically Sorted Source Nodes: [x_250, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        buf116 = extern_kernels.convolution(buf115, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg194_1
        del buf115
        buf117 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_250, x_se_48, x_se_49, x_se_50, x_se_51, hardsigmoid_12, x_251], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_40.run(buf117, buf116, arg195_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg195_1
        # Topologically Sorted Source Nodes: [x_250, x_se_48, x_se_49, x_se_50, x_se_51, hardsigmoid_12, x_251, x_252], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf118 = extern_kernels.convolution(buf117, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg196_1
        buf119 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_41.run(buf119, buf118, arg197_1, arg198_1, arg199_1, arg200_1, 175616, grid=grid(175616), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        del buf118
        # Topologically Sorted Source Nodes: [x_253, x_254, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg201_1
        del buf119
        buf121 = buf120; del buf120  # reuse
        buf122 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_35.run(buf121, arg202_1, arg203_1, arg204_1, arg205_1, buf122, 1053696, grid=grid(1053696), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        del buf121
        # Topologically Sorted Source Nodes: [x_257, x_258], Original ATen: [aten.hardswish, aten.convolution]
        buf123 = extern_kernels.convolution(buf122, arg206_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf123, (8, 672, 7, 7), (32928, 1, 4704, 672))
        del arg206_1
        del buf122
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_42.run(buf124, arg207_1, arg208_1, arg209_1, arg210_1, 263424, grid=grid(263424), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        buf126 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_260, x_se_52], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_43.run(buf124, buf126, 5376, 49, grid=grid(5376), stream=stream0)
        # Topologically Sorted Source Nodes: [x_260, x_se_52, x_se_53], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf127 = extern_kernels.convolution(buf126, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 168, 1, 1), (168, 1, 1, 1))
        del arg211_1
        del buf126
        buf128 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_260, x_se_52, x_se_53, x_se_54], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_39.run(buf128, arg212_1, 1344, grid=grid(1344), stream=stream0)
        del arg212_1
        # Topologically Sorted Source Nodes: [x_260, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        buf129 = extern_kernels.convolution(buf128, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg213_1
        del buf128
        buf130 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_260, x_se_52, x_se_53, x_se_54, x_se_55, hardsigmoid_13, x_261], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_44.run(buf130, buf129, arg214_1, 263424, grid=grid(263424), stream=stream0)
        del arg214_1
        del buf129
        # Topologically Sorted Source Nodes: [x_260, x_se_52, x_se_53, x_se_54, x_se_55, hardsigmoid_13, x_261, x_262], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf131 = extern_kernels.convolution(buf130, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 160, 7, 7), (7840, 1, 1120, 160))
        del arg215_1
        del buf130
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_263], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_45.run(buf132, arg216_1, arg217_1, arg218_1, arg219_1, 62720, grid=grid(62720), stream=stream0)
        del arg216_1
        del arg217_1
        del arg218_1
        del arg219_1
        # Topologically Sorted Source Nodes: [x_264], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del arg220_1
        buf134 = buf133; del buf133  # reuse
        buf135 = reinterpret_tensor(buf65, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_265, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_46.run(buf134, arg221_1, arg222_1, arg223_1, arg224_1, buf135, 376320, grid=grid(376320), stream=stream0)
        del arg221_1
        del arg222_1
        del arg223_1
        del arg224_1
        del buf134
        # Topologically Sorted Source Nodes: [x_266, x_267], Original ATen: [aten.hardswish, aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg225_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf136, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del arg225_1
        del buf135
        buf137 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_268], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_47.run(buf137, arg226_1, arg227_1, arg228_1, arg229_1, 376320, grid=grid(376320), stream=stream0)
        del arg226_1
        del arg227_1
        del arg228_1
        del arg229_1
        buf139 = reinterpret_tensor(buf97, (8, 960, 1, 1), (960, 1, 1, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_269, x_se_56], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_48.run(buf137, buf139, 7680, 49, grid=grid(7680), stream=stream0)
        # Topologically Sorted Source Nodes: [x_269, x_se_56, x_se_57], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf140 = extern_kernels.convolution(buf139, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg230_1
        del buf139
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_269, x_se_56, x_se_57, x_se_58], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_49.run(buf141, arg231_1, 1920, grid=grid(1920), stream=stream0)
        del arg231_1
        # Topologically Sorted Source Nodes: [x_269, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        buf142 = extern_kernels.convolution(buf141, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 960, 1, 1), (960, 1, 1, 1))
        del arg232_1
        del buf141
        buf143 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [x_269, x_se_56, x_se_57, x_se_58, x_se_59, hardsigmoid_14, x_270], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_50.run(buf143, buf142, arg233_1, 376320, grid=grid(376320), stream=stream0)
        del arg233_1
        # Topologically Sorted Source Nodes: [x_269, x_se_56, x_se_57, x_se_58, x_se_59, hardsigmoid_14, x_270, x_271], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf144 = extern_kernels.convolution(buf143, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (8, 160, 7, 7), (7840, 1, 1120, 160))
        del arg234_1
        buf145 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_51.run(buf145, buf144, arg235_1, arg236_1, arg237_1, arg238_1, 62720, grid=grid(62720), stream=stream0)
        del arg235_1
        del arg236_1
        del arg237_1
        del arg238_1
        del buf144
        # Topologically Sorted Source Nodes: [x_274], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del arg239_1
        buf147 = buf146; del buf146  # reuse
        buf148 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_275, x_276], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_46.run(buf147, arg240_1, arg241_1, arg242_1, arg243_1, buf148, 376320, grid=grid(376320), stream=stream0)
        del arg240_1
        del arg241_1
        del arg242_1
        del arg243_1
        del buf147
        # Topologically Sorted Source Nodes: [x_276, x_277], Original ATen: [aten.hardswish, aten.convolution]
        buf149 = extern_kernels.convolution(buf148, arg244_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf149, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del arg244_1
        del buf148
        buf150 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_47.run(buf150, arg245_1, arg246_1, arg247_1, arg248_1, 376320, grid=grid(376320), stream=stream0)
        del arg245_1
        del arg246_1
        del arg247_1
        del arg248_1
        buf152 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [x_279, x_se_60], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_48.run(buf150, buf152, 7680, 49, grid=grid(7680), stream=stream0)
        # Topologically Sorted Source Nodes: [x_279, x_se_60, x_se_61], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf153 = extern_kernels.convolution(buf152, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg249_1
        del buf152
        buf154 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [x_279, x_se_60, x_se_61, x_se_62], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_49.run(buf154, arg250_1, 1920, grid=grid(1920), stream=stream0)
        del arg250_1
        # Topologically Sorted Source Nodes: [x_279, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        buf155 = extern_kernels.convolution(buf154, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (8, 960, 1, 1), (960, 1, 1, 1))
        del arg251_1
        del buf154
        buf156 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [x_279, x_se_60, x_se_61, x_se_62, x_se_63, hardsigmoid_15, x_280], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_50.run(buf156, buf155, arg252_1, 376320, grid=grid(376320), stream=stream0)
        del arg252_1
        # Topologically Sorted Source Nodes: [x_279, x_se_60, x_se_61, x_se_62, x_se_63, hardsigmoid_15, x_280, x_281], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf157 = extern_kernels.convolution(buf156, arg253_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 160, 7, 7), (7840, 1, 1120, 160))
        del arg253_1
        del buf156
        buf158 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [x_282, x_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_51.run(buf158, buf157, arg254_1, arg255_1, arg256_1, arg257_1, 62720, grid=grid(62720), stream=stream0)
        del arg254_1
        del arg255_1
        del arg256_1
        del arg257_1
        del buf157
        # Topologically Sorted Source Nodes: [x_282, x_283, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf159 = extern_kernels.convolution(buf158, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del arg258_1
        del buf158
        buf160 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_47.run(buf160, arg259_1, arg260_1, arg261_1, arg262_1, 376320, grid=grid(376320), stream=stream0)
        del arg259_1
        del arg260_1
        del arg261_1
        del arg262_1
        buf162 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_286, x_287], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_48.run(buf160, buf162, 7680, 49, grid=grid(7680), stream=stream0)
        del buf160
        # Topologically Sorted Source Nodes: [x_286, x_287, x_288], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf163 = extern_kernels.convolution(buf162, arg263_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 1280, 1, 1), (1280, 1, 1, 1))
        del arg263_1
        del buf162
        buf164 = reinterpret_tensor(buf163, (8, 1280, 1, 1), (1280, 1, 10240, 10240), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_286, x_287, x_288, x_289], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_52.run(buf164, arg264_1, 10240, grid=grid(10240), stream=stream0)
        del arg264_1
        buf165 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg266_1, reinterpret_tensor(buf164, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg265_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf165)
        del arg265_1
        del arg266_1
        del buf164
    return (buf165, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((200, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((200, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((80, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((672, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((160, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1280, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenetv3_large_100', benchmark_compiled_module)
