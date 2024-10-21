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
# Topologically Sorted Source Nodes: [x_192], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_192 => convolution_65
# Graph fragment:
#   %convolution_65 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
# Topologically Sorted Source Nodes: [x_192], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_192 => convolution_65
# Graph fragment:
#   %convolution_65 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/s2/cs2jw6ww4qqadf4qzhfgqg4ytskum3mmkv7uf57gqxjogkkzy2de.py
# Topologically Sorted Source Nodes: [x_193, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_193 => add_146, mul_196, mul_197, sub_65
#   x_194 => relu_44
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_65, %unsqueeze_521), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_196, %unsqueeze_525), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_197, %unsqueeze_527), kwargs = {})
#   %relu_44 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_146,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bn/cbnrljvnwib6vqg3u22o5iek3lllnkbtlz44t2fulehcypefyqpy.py
# Topologically Sorted Source Nodes: [x_202, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_202 => add_152, mul_205, mul_206, sub_68
#   x_203 => add_153
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_545), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_205, %unsqueeze_549), kwargs = {})
#   %add_152 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_206, %unsqueeze_551), kwargs = {})
#   %add_153 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_152, %relu_44), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2d/c2dki33hpxebrsithuuvlr54vawbb5etfx2dol5vu5dtgzyduewb.py
# Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_205 => add_155, mul_208, mul_209, sub_69
#   x_206 => relu_47
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_69, %unsqueeze_553), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_208, %unsqueeze_557), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_209, %unsqueeze_559), kwargs = {})
#   %relu_47 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_155,), kwargs = {})
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
    xnumel = 9633792
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rr/crrko7v62ggmnd7zes45hssoqvrwhprrdkralflsqgzblt2cbrln.py
# Topologically Sorted Source Nodes: [x_208, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_208 => add_157, mul_211, mul_212, sub_70
#   x_209 => relu_48
# Graph fragment:
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_70, %unsqueeze_561), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_563), kwargs = {})
#   %mul_212 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_211, %unsqueeze_565), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_212, %unsqueeze_567), kwargs = {})
#   %relu_48 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_157,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6q/c6qq63q2pcgdwvdvk64cosehcntbxzanyaihjq3dvq4es7whis32.py
# Topologically Sorted Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_211 => add_159, mul_214, mul_215, sub_71
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_569), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_571), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_214, %unsqueeze_573), kwargs = {})
#   %add_159 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_215, %unsqueeze_575), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qc/cqckvlme6s5xpf4rp5uwlpb7ywmbzahmljlj7q6jhgysjlw5igba.py
# Topologically Sorted Source Nodes: [x_213, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_213 => add_161, mul_217, mul_218, sub_72
#   x_214 => relu_49
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_72, %unsqueeze_577), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_217, %unsqueeze_581), kwargs = {})
#   %add_161 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_218, %unsqueeze_583), kwargs = {})
#   %relu_49 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_161,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d6/cd6576ysi53fuppsa6ll5qtdgbkmpn6tkizdrsifwwmwp5e7uwfg.py
# Topologically Sorted Source Nodes: [x_219, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_219 => add_165, mul_223, mul_224, sub_74
#   x_220 => add_166
# Graph fragment:
#   %sub_74 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_74, %unsqueeze_593), kwargs = {})
#   %mul_223 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_74, %unsqueeze_595), kwargs = {})
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_223, %unsqueeze_597), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_224, %unsqueeze_599), kwargs = {})
#   %add_166 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_165, %add_159), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pt/cptq3pb4edxqdogvcf55spxgvg6db5xnd5aofgsiayaxw55cvwsd.py
# Topologically Sorted Source Nodes: [x_231, x_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_231 => add_175, mul_235, mul_236, sub_78
#   x_232 => relu_53
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_78, %unsqueeze_625), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_235, %unsqueeze_629), kwargs = {})
#   %add_175 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_236, %unsqueeze_631), kwargs = {})
#   %relu_53 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_175,), kwargs = {})
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
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 144
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/po/cpopowqa6ho3j6zq2fvvwmltno6iujntqdkzpaotc2uyspwrpgfb.py
# Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_234 => add_177, mul_238, mul_239, sub_79
#   x_235 => relu_54
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_79, %unsqueeze_633), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_635), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_238, %unsqueeze_637), kwargs = {})
#   %add_177 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_239, %unsqueeze_639), kwargs = {})
#   %relu_54 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_177,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
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


# kernel path: /tmp/torchinductor_sahanp/ks/cksldde2mq44dgzsvffe6ohyqo5h7ig3qjd7fkzrsaqp57cacsto.py
# Topologically Sorted Source Nodes: [x_237], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_237 => add_179, mul_241, mul_242, sub_80
# Graph fragment:
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_80, %unsqueeze_641), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_643), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_241, %unsqueeze_645), kwargs = {})
#   %add_179 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_242, %unsqueeze_647), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5s/c5sz5wghhvas2a3l7us3m7qb75ozutptfgpbgguucrnjw3hfdaku.py
# Topologically Sorted Source Nodes: [x_239, x_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_239 => add_181, mul_244, mul_245, sub_81
#   x_240 => relu_55
# Graph fragment:
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_81, %unsqueeze_649), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_651), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_244, %unsqueeze_653), kwargs = {})
#   %add_181 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_245, %unsqueeze_655), kwargs = {})
#   %relu_55 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_181,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wv/cwv4hy4pjkmo3i3jhmisbausryeunafktb6svvjiwwrzvslujnar.py
# Topologically Sorted Source Nodes: [x_245, x_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_245 => add_185, mul_250, mul_251, sub_83
#   x_246 => add_186
# Graph fragment:
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_83, %unsqueeze_665), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_667), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_250, %unsqueeze_669), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_251, %unsqueeze_671), kwargs = {})
#   %add_186 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_185, %add_179), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
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
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/it/citk6w2uztr7oo6axpixyycp7blsdxe5pfdijhvd6iceqap34gbd.py
# Topologically Sorted Source Nodes: [x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_248 => add_188, mul_253, mul_254, sub_84
#   x_249 => relu_57
# Graph fragment:
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_84, %unsqueeze_673), kwargs = {})
#   %mul_253 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_675), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_253, %unsqueeze_677), kwargs = {})
#   %add_188 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_254, %unsqueeze_679), kwargs = {})
#   %relu_57 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_188,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kz/ckzjrx54433iynnt3ynwc4wtzgg4du7ig5jan3nab4644sky2ypy.py
# Topologically Sorted Source Nodes: [x_269, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_269 => add_204, mul_274, mul_275, sub_91
#   x_270 => relu_62
# Graph fragment:
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_91, %unsqueeze_729), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %unsqueeze_731), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_274, %unsqueeze_733), kwargs = {})
#   %add_204 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_275, %unsqueeze_735), kwargs = {})
#   %relu_62 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_204,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
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


# kernel path: /tmp/torchinductor_sahanp/5m/c5m5p3hvfuyzifictvah25cwac5s34o52vsajewmtakztirgyqbx.py
# Topologically Sorted Source Nodes: [x_272], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_272 => add_206, mul_277, mul_278, sub_92
# Graph fragment:
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_92, %unsqueeze_737), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_277, %unsqueeze_741), kwargs = {})
#   %add_206 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_278, %unsqueeze_743), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
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


# kernel path: /tmp/torchinductor_sahanp/3a/c3ayk3e7mmqnoveyuvmyvdzifhhgf4agjo6iausebk3lvdwrumti.py
# Topologically Sorted Source Nodes: [x_280, x_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_280 => add_212, mul_286, mul_287, sub_95
#   x_281 => add_213
# Graph fragment:
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_95, %unsqueeze_761), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %unsqueeze_763), kwargs = {})
#   %mul_287 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_286, %unsqueeze_765), kwargs = {})
#   %add_212 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_287, %unsqueeze_767), kwargs = {})
#   %add_213 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_212, %add_206), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ey/cey2myw3idkf5cne4fag437zocbluhhppfntbyxj7m5sldd6axpk.py
# Topologically Sorted Source Nodes: [x_283, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_283 => add_215, mul_289, mul_290, sub_96
#   x_284 => relu_65
# Graph fragment:
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_96, %unsqueeze_769), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %unsqueeze_771), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_773), kwargs = {})
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %unsqueeze_775), kwargs = {})
#   %relu_65 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_215,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2d/c2dcnqppn2lvb2njzztmyrrgsqwktdhio4k3p3mciifrussw4rdu.py
# Topologically Sorted Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_307 => add_233, mul_313, mul_314, sub_104
# Graph fragment:
#   %sub_104 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_104, %unsqueeze_833), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_104, %unsqueeze_835), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_313, %unsqueeze_837), kwargs = {})
#   %add_233 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_314, %unsqueeze_839), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/c7/cc7chfitecjv2ezu32sv5f3ivymktpy7pz5b2gio7adsbbg4yaur.py
# Topologically Sorted Source Nodes: [x_309, x_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_309 => add_235, mul_316, mul_317, sub_105
#   x_310 => relu_71
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_841), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_316, %unsqueeze_845), kwargs = {})
#   %add_235 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_317, %unsqueeze_847), kwargs = {})
#   %relu_71 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_235,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/me/cmecxyw5ut35lqbuwafhcdv4jdn5wmfe3tb3x2lk32bgpaannuxh.py
# Topologically Sorted Source Nodes: [x_315, x_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_315 => add_239, mul_322, mul_323, sub_107
#   x_316 => add_240
# Graph fragment:
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_107, %unsqueeze_857), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %unsqueeze_859), kwargs = {})
#   %mul_323 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_322, %unsqueeze_861), kwargs = {})
#   %add_239 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_323, %unsqueeze_863), kwargs = {})
#   %add_240 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_239, %add_233), kwargs = {})
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
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ow/cowczvpah77caimcosrpds3guivgc7dsje3phc4mjrikb4plzaui.py
# Topologically Sorted Source Nodes: [x_327, x_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_327 => add_249, mul_334, mul_335, sub_111
#   x_328 => relu_75
# Graph fragment:
#   %sub_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_111, %unsqueeze_889), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_111, %unsqueeze_891), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_334, %unsqueeze_893), kwargs = {})
#   %add_249 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_335, %unsqueeze_895), kwargs = {})
#   %relu_75 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_249,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 336
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


# kernel path: /tmp/torchinductor_sahanp/5m/c5mdf2x4wgv62phxnlzjlp2lr4c3vwngyspfbca77bhahyw3volf.py
# Topologically Sorted Source Nodes: [x_339, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_339 => add_258, mul_346, mul_347, sub_115
#   x_340 => relu_78
# Graph fragment:
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_921), kwargs = {})
#   %mul_346 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %unsqueeze_923), kwargs = {})
#   %mul_347 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_346, %unsqueeze_925), kwargs = {})
#   %add_258 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_347, %unsqueeze_927), kwargs = {})
#   %relu_78 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_258,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/mn/cmnbbwairsxwpukvxjzfd2u22lwdyy3qs4mntmjtbk6ai4thpwlp.py
# Topologically Sorted Source Nodes: [x_342], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_342 => add_260, mul_349, mul_350, sub_116
# Graph fragment:
#   %sub_116 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_116, %unsqueeze_929), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_116, %unsqueeze_931), kwargs = {})
#   %mul_350 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_349, %unsqueeze_933), kwargs = {})
#   %add_260 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_350, %unsqueeze_935), kwargs = {})
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
    xnumel = 72128
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


# kernel path: /tmp/torchinductor_sahanp/qz/cqzkdltrxofj7p3rt5mhitg347wx4jpn6734xe2fxjv2tmqxgx7p.py
# Topologically Sorted Source Nodes: [x_344, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_344 => add_262, mul_352, mul_353, sub_117
#   x_345 => relu_79
# Graph fragment:
#   %sub_117 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_117, %unsqueeze_937), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_117, %unsqueeze_939), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, %unsqueeze_941), kwargs = {})
#   %add_262 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_353, %unsqueeze_943), kwargs = {})
#   %relu_79 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_262,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 432768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1104
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


# kernel path: /tmp/torchinductor_sahanp/6y/c6ygl56hqlkcvedguiahcg57csrkhzaml3qqvwnr4vij6rpxxiyk.py
# Topologically Sorted Source Nodes: [x_350, x_351], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_350 => add_266, mul_358, mul_359, sub_119
#   x_351 => add_267
# Graph fragment:
#   %sub_119 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_119, %unsqueeze_953), kwargs = {})
#   %mul_358 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_119, %unsqueeze_955), kwargs = {})
#   %mul_359 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_358, %unsqueeze_957), kwargs = {})
#   %add_266 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_359, %unsqueeze_959), kwargs = {})
#   %add_267 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_266, %add_260), kwargs = {})
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
    xnumel = 72128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 184
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eo/ceoftkgz3465ka37nph4hml5a3dnpds5ll6j3u4ktea7vgcfo4eh.py
# Topologically Sorted Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_377 => add_287, mul_385, mul_386, sub_128
# Graph fragment:
#   %sub_128 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_128, %unsqueeze_1025), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_128, %unsqueeze_1027), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_385, %unsqueeze_1029), kwargs = {})
#   %add_287 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_386, %unsqueeze_1031), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 137984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 352
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


# kernel path: /tmp/torchinductor_sahanp/lq/clqbgg5pqo3bdnkju24x6bbklzkiro7tz5c4lzojbl43cgzynxic.py
# Topologically Sorted Source Nodes: [x_379, x_380, x_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   x_379 => add_289, mul_388, mul_389, sub_129
#   x_380 => relu_87
#   x_381 => mean_1
# Graph fragment:
#   %sub_129 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_129, %unsqueeze_1033), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %unsqueeze_1035), kwargs = {})
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_388, %unsqueeze_1037), kwargs = {})
#   %add_289 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_389, %unsqueeze_1039), kwargs = {})
#   %relu_87 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_289,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_87, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 15872
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1984
    x1 = (xindex // 1984)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1984*r2) + (97216*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 49.0
    tmp23 = tmp21 / tmp22
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (16, ), (1, ))
    assert_size_stride(arg9_1, (16, ), (1, ))
    assert_size_stride(arg10_1, (16, ), (1, ))
    assert_size_stride(arg11_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg12_1, (16, ), (1, ))
    assert_size_stride(arg13_1, (16, ), (1, ))
    assert_size_stride(arg14_1, (16, ), (1, ))
    assert_size_stride(arg15_1, (16, ), (1, ))
    assert_size_stride(arg16_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg17_1, (16, ), (1, ))
    assert_size_stride(arg18_1, (16, ), (1, ))
    assert_size_stride(arg19_1, (16, ), (1, ))
    assert_size_stride(arg20_1, (16, ), (1, ))
    assert_size_stride(arg21_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg22_1, (96, ), (1, ))
    assert_size_stride(arg23_1, (96, ), (1, ))
    assert_size_stride(arg24_1, (96, ), (1, ))
    assert_size_stride(arg25_1, (96, ), (1, ))
    assert_size_stride(arg26_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg27_1, (96, ), (1, ))
    assert_size_stride(arg28_1, (96, ), (1, ))
    assert_size_stride(arg29_1, (96, ), (1, ))
    assert_size_stride(arg30_1, (96, ), (1, ))
    assert_size_stride(arg31_1, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg32_1, (24, ), (1, ))
    assert_size_stride(arg33_1, (24, ), (1, ))
    assert_size_stride(arg34_1, (24, ), (1, ))
    assert_size_stride(arg35_1, (24, ), (1, ))
    assert_size_stride(arg36_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg37_1, (24, ), (1, ))
    assert_size_stride(arg38_1, (24, ), (1, ))
    assert_size_stride(arg39_1, (24, ), (1, ))
    assert_size_stride(arg40_1, (24, ), (1, ))
    assert_size_stride(arg41_1, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg42_1, (24, ), (1, ))
    assert_size_stride(arg43_1, (24, ), (1, ))
    assert_size_stride(arg44_1, (24, ), (1, ))
    assert_size_stride(arg45_1, (24, ), (1, ))
    assert_size_stride(arg46_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg47_1, (24, ), (1, ))
    assert_size_stride(arg48_1, (24, ), (1, ))
    assert_size_stride(arg49_1, (24, ), (1, ))
    assert_size_stride(arg50_1, (24, ), (1, ))
    assert_size_stride(arg51_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg52_1, (24, ), (1, ))
    assert_size_stride(arg53_1, (24, ), (1, ))
    assert_size_stride(arg54_1, (24, ), (1, ))
    assert_size_stride(arg55_1, (24, ), (1, ))
    assert_size_stride(arg56_1, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg57_1, (24, ), (1, ))
    assert_size_stride(arg58_1, (24, ), (1, ))
    assert_size_stride(arg59_1, (24, ), (1, ))
    assert_size_stride(arg60_1, (24, ), (1, ))
    assert_size_stride(arg61_1, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg62_1, (24, ), (1, ))
    assert_size_stride(arg63_1, (24, ), (1, ))
    assert_size_stride(arg64_1, (24, ), (1, ))
    assert_size_stride(arg65_1, (24, ), (1, ))
    assert_size_stride(arg66_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg67_1, (144, ), (1, ))
    assert_size_stride(arg68_1, (144, ), (1, ))
    assert_size_stride(arg69_1, (144, ), (1, ))
    assert_size_stride(arg70_1, (144, ), (1, ))
    assert_size_stride(arg71_1, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg72_1, (144, ), (1, ))
    assert_size_stride(arg73_1, (144, ), (1, ))
    assert_size_stride(arg74_1, (144, ), (1, ))
    assert_size_stride(arg75_1, (144, ), (1, ))
    assert_size_stride(arg76_1, (32, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg77_1, (32, ), (1, ))
    assert_size_stride(arg78_1, (32, ), (1, ))
    assert_size_stride(arg79_1, (32, ), (1, ))
    assert_size_stride(arg80_1, (32, ), (1, ))
    assert_size_stride(arg81_1, (96, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg82_1, (96, ), (1, ))
    assert_size_stride(arg83_1, (96, ), (1, ))
    assert_size_stride(arg84_1, (96, ), (1, ))
    assert_size_stride(arg85_1, (96, ), (1, ))
    assert_size_stride(arg86_1, (96, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg87_1, (96, ), (1, ))
    assert_size_stride(arg88_1, (96, ), (1, ))
    assert_size_stride(arg89_1, (96, ), (1, ))
    assert_size_stride(arg90_1, (96, ), (1, ))
    assert_size_stride(arg91_1, (32, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg92_1, (32, ), (1, ))
    assert_size_stride(arg93_1, (32, ), (1, ))
    assert_size_stride(arg94_1, (32, ), (1, ))
    assert_size_stride(arg95_1, (32, ), (1, ))
    assert_size_stride(arg96_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg97_1, (192, ), (1, ))
    assert_size_stride(arg98_1, (192, ), (1, ))
    assert_size_stride(arg99_1, (192, ), (1, ))
    assert_size_stride(arg100_1, (192, ), (1, ))
    assert_size_stride(arg101_1, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg102_1, (192, ), (1, ))
    assert_size_stride(arg103_1, (192, ), (1, ))
    assert_size_stride(arg104_1, (192, ), (1, ))
    assert_size_stride(arg105_1, (192, ), (1, ))
    assert_size_stride(arg106_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg107_1, (32, ), (1, ))
    assert_size_stride(arg108_1, (32, ), (1, ))
    assert_size_stride(arg109_1, (32, ), (1, ))
    assert_size_stride(arg110_1, (32, ), (1, ))
    assert_size_stride(arg111_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg112_1, (192, ), (1, ))
    assert_size_stride(arg113_1, (192, ), (1, ))
    assert_size_stride(arg114_1, (192, ), (1, ))
    assert_size_stride(arg115_1, (192, ), (1, ))
    assert_size_stride(arg116_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg117_1, (192, ), (1, ))
    assert_size_stride(arg118_1, (192, ), (1, ))
    assert_size_stride(arg119_1, (192, ), (1, ))
    assert_size_stride(arg120_1, (192, ), (1, ))
    assert_size_stride(arg121_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg122_1, (32, ), (1, ))
    assert_size_stride(arg123_1, (32, ), (1, ))
    assert_size_stride(arg124_1, (32, ), (1, ))
    assert_size_stride(arg125_1, (32, ), (1, ))
    assert_size_stride(arg126_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg127_1, (192, ), (1, ))
    assert_size_stride(arg128_1, (192, ), (1, ))
    assert_size_stride(arg129_1, (192, ), (1, ))
    assert_size_stride(arg130_1, (192, ), (1, ))
    assert_size_stride(arg131_1, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg132_1, (192, ), (1, ))
    assert_size_stride(arg133_1, (192, ), (1, ))
    assert_size_stride(arg134_1, (192, ), (1, ))
    assert_size_stride(arg135_1, (192, ), (1, ))
    assert_size_stride(arg136_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg137_1, (64, ), (1, ))
    assert_size_stride(arg138_1, (64, ), (1, ))
    assert_size_stride(arg139_1, (64, ), (1, ))
    assert_size_stride(arg140_1, (64, ), (1, ))
    assert_size_stride(arg141_1, (192, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg142_1, (192, ), (1, ))
    assert_size_stride(arg143_1, (192, ), (1, ))
    assert_size_stride(arg144_1, (192, ), (1, ))
    assert_size_stride(arg145_1, (192, ), (1, ))
    assert_size_stride(arg146_1, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg147_1, (192, ), (1, ))
    assert_size_stride(arg148_1, (192, ), (1, ))
    assert_size_stride(arg149_1, (192, ), (1, ))
    assert_size_stride(arg150_1, (192, ), (1, ))
    assert_size_stride(arg151_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg152_1, (64, ), (1, ))
    assert_size_stride(arg153_1, (64, ), (1, ))
    assert_size_stride(arg154_1, (64, ), (1, ))
    assert_size_stride(arg155_1, (64, ), (1, ))
    assert_size_stride(arg156_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg157_1, (384, ), (1, ))
    assert_size_stride(arg158_1, (384, ), (1, ))
    assert_size_stride(arg159_1, (384, ), (1, ))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (384, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg162_1, (384, ), (1, ))
    assert_size_stride(arg163_1, (384, ), (1, ))
    assert_size_stride(arg164_1, (384, ), (1, ))
    assert_size_stride(arg165_1, (384, ), (1, ))
    assert_size_stride(arg166_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg167_1, (64, ), (1, ))
    assert_size_stride(arg168_1, (64, ), (1, ))
    assert_size_stride(arg169_1, (64, ), (1, ))
    assert_size_stride(arg170_1, (64, ), (1, ))
    assert_size_stride(arg171_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (384, ), (1, ))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (384, ), (1, ))
    assert_size_stride(arg176_1, (384, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (384, ), (1, ))
    assert_size_stride(arg179_1, (384, ), (1, ))
    assert_size_stride(arg180_1, (384, ), (1, ))
    assert_size_stride(arg181_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg182_1, (64, ), (1, ))
    assert_size_stride(arg183_1, (64, ), (1, ))
    assert_size_stride(arg184_1, (64, ), (1, ))
    assert_size_stride(arg185_1, (64, ), (1, ))
    assert_size_stride(arg186_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg187_1, (384, ), (1, ))
    assert_size_stride(arg188_1, (384, ), (1, ))
    assert_size_stride(arg189_1, (384, ), (1, ))
    assert_size_stride(arg190_1, (384, ), (1, ))
    assert_size_stride(arg191_1, (384, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg192_1, (384, ), (1, ))
    assert_size_stride(arg193_1, (384, ), (1, ))
    assert_size_stride(arg194_1, (384, ), (1, ))
    assert_size_stride(arg195_1, (384, ), (1, ))
    assert_size_stride(arg196_1, (112, 384, 1, 1), (384, 1, 1, 1))
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
    assert_size_stride(arg211_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg212_1, (112, ), (1, ))
    assert_size_stride(arg213_1, (112, ), (1, ))
    assert_size_stride(arg214_1, (112, ), (1, ))
    assert_size_stride(arg215_1, (112, ), (1, ))
    assert_size_stride(arg216_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg217_1, (672, ), (1, ))
    assert_size_stride(arg218_1, (672, ), (1, ))
    assert_size_stride(arg219_1, (672, ), (1, ))
    assert_size_stride(arg220_1, (672, ), (1, ))
    assert_size_stride(arg221_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg222_1, (672, ), (1, ))
    assert_size_stride(arg223_1, (672, ), (1, ))
    assert_size_stride(arg224_1, (672, ), (1, ))
    assert_size_stride(arg225_1, (672, ), (1, ))
    assert_size_stride(arg226_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg227_1, (112, ), (1, ))
    assert_size_stride(arg228_1, (112, ), (1, ))
    assert_size_stride(arg229_1, (112, ), (1, ))
    assert_size_stride(arg230_1, (112, ), (1, ))
    assert_size_stride(arg231_1, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg232_1, (336, ), (1, ))
    assert_size_stride(arg233_1, (336, ), (1, ))
    assert_size_stride(arg234_1, (336, ), (1, ))
    assert_size_stride(arg235_1, (336, ), (1, ))
    assert_size_stride(arg236_1, (336, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg237_1, (336, ), (1, ))
    assert_size_stride(arg238_1, (336, ), (1, ))
    assert_size_stride(arg239_1, (336, ), (1, ))
    assert_size_stride(arg240_1, (336, ), (1, ))
    assert_size_stride(arg241_1, (112, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg242_1, (112, ), (1, ))
    assert_size_stride(arg243_1, (112, ), (1, ))
    assert_size_stride(arg244_1, (112, ), (1, ))
    assert_size_stride(arg245_1, (112, ), (1, ))
    assert_size_stride(arg246_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg247_1, (672, ), (1, ))
    assert_size_stride(arg248_1, (672, ), (1, ))
    assert_size_stride(arg249_1, (672, ), (1, ))
    assert_size_stride(arg250_1, (672, ), (1, ))
    assert_size_stride(arg251_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg252_1, (672, ), (1, ))
    assert_size_stride(arg253_1, (672, ), (1, ))
    assert_size_stride(arg254_1, (672, ), (1, ))
    assert_size_stride(arg255_1, (672, ), (1, ))
    assert_size_stride(arg256_1, (184, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg257_1, (184, ), (1, ))
    assert_size_stride(arg258_1, (184, ), (1, ))
    assert_size_stride(arg259_1, (184, ), (1, ))
    assert_size_stride(arg260_1, (184, ), (1, ))
    assert_size_stride(arg261_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg262_1, (1104, ), (1, ))
    assert_size_stride(arg263_1, (1104, ), (1, ))
    assert_size_stride(arg264_1, (1104, ), (1, ))
    assert_size_stride(arg265_1, (1104, ), (1, ))
    assert_size_stride(arg266_1, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg267_1, (1104, ), (1, ))
    assert_size_stride(arg268_1, (1104, ), (1, ))
    assert_size_stride(arg269_1, (1104, ), (1, ))
    assert_size_stride(arg270_1, (1104, ), (1, ))
    assert_size_stride(arg271_1, (184, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg272_1, (184, ), (1, ))
    assert_size_stride(arg273_1, (184, ), (1, ))
    assert_size_stride(arg274_1, (184, ), (1, ))
    assert_size_stride(arg275_1, (184, ), (1, ))
    assert_size_stride(arg276_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg277_1, (1104, ), (1, ))
    assert_size_stride(arg278_1, (1104, ), (1, ))
    assert_size_stride(arg279_1, (1104, ), (1, ))
    assert_size_stride(arg280_1, (1104, ), (1, ))
    assert_size_stride(arg281_1, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg282_1, (1104, ), (1, ))
    assert_size_stride(arg283_1, (1104, ), (1, ))
    assert_size_stride(arg284_1, (1104, ), (1, ))
    assert_size_stride(arg285_1, (1104, ), (1, ))
    assert_size_stride(arg286_1, (184, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg287_1, (184, ), (1, ))
    assert_size_stride(arg288_1, (184, ), (1, ))
    assert_size_stride(arg289_1, (184, ), (1, ))
    assert_size_stride(arg290_1, (184, ), (1, ))
    assert_size_stride(arg291_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg292_1, (1104, ), (1, ))
    assert_size_stride(arg293_1, (1104, ), (1, ))
    assert_size_stride(arg294_1, (1104, ), (1, ))
    assert_size_stride(arg295_1, (1104, ), (1, ))
    assert_size_stride(arg296_1, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg297_1, (1104, ), (1, ))
    assert_size_stride(arg298_1, (1104, ), (1, ))
    assert_size_stride(arg299_1, (1104, ), (1, ))
    assert_size_stride(arg300_1, (1104, ), (1, ))
    assert_size_stride(arg301_1, (184, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg302_1, (184, ), (1, ))
    assert_size_stride(arg303_1, (184, ), (1, ))
    assert_size_stride(arg304_1, (184, ), (1, ))
    assert_size_stride(arg305_1, (184, ), (1, ))
    assert_size_stride(arg306_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg307_1, (1104, ), (1, ))
    assert_size_stride(arg308_1, (1104, ), (1, ))
    assert_size_stride(arg309_1, (1104, ), (1, ))
    assert_size_stride(arg310_1, (1104, ), (1, ))
    assert_size_stride(arg311_1, (1104, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg312_1, (1104, ), (1, ))
    assert_size_stride(arg313_1, (1104, ), (1, ))
    assert_size_stride(arg314_1, (1104, ), (1, ))
    assert_size_stride(arg315_1, (1104, ), (1, ))
    assert_size_stride(arg316_1, (352, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg317_1, (352, ), (1, ))
    assert_size_stride(arg318_1, (352, ), (1, ))
    assert_size_stride(arg319_1, (352, ), (1, ))
    assert_size_stride(arg320_1, (352, ), (1, ))
    assert_size_stride(arg321_1, (1984, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(arg322_1, (1984, ), (1, ))
    assert_size_stride(arg323_1, (1984, ), (1, ))
    assert_size_stride(arg324_1, (1984, ), (1, ))
    assert_size_stride(arg325_1, (1984, ), (1, ))
    assert_size_stride(arg326_1, (1000, 1984), (1984, 1))
    assert_size_stride(arg327_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_192], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((16, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_192], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 48, 9, grid=grid(48, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_192], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_193, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        # Topologically Sorted Source Nodes: [x_195], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del arg6_1
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_196, x_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf5, arg7_1, arg8_1, arg9_1, arg10_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [x_196, x_197, x_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg11_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf6, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del arg11_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_199, x_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf7, arg12_1, arg13_1, arg14_1, arg15_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        # Topologically Sorted Source Nodes: [x_199, x_200, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del arg16_1
        del buf7
        buf9 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_202, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_3.run(buf9, buf8, arg17_1, arg18_1, arg19_1, arg20_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del buf8
        # Topologically Sorted Source Nodes: [x_202, x_203, x_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 96, 112, 112), (1204224, 1, 10752, 96))
        del arg21_1
        del buf9
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf11, arg22_1, arg23_1, arg24_1, arg25_1, 9633792, grid=grid(9633792), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        # Topologically Sorted Source Nodes: [x_205, x_206, x_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg26_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf12, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del arg26_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_208, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf13, arg27_1, arg28_1, arg29_1, arg30_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        # Topologically Sorted Source Nodes: [x_208, x_209, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg31_1
        del buf13
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf15, arg32_1, arg33_1, arg34_1, arg35_1, 602112, grid=grid(602112), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        # Topologically Sorted Source Nodes: [x_212], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg36_1
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_213, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf17, arg37_1, arg38_1, arg39_1, arg40_1, 602112, grid=grid(602112), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        # Topologically Sorted Source Nodes: [x_213, x_214, x_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg41_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf18, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg41_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_216, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf19, arg42_1, arg43_1, arg44_1, arg45_1, 602112, grid=grid(602112), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        # Topologically Sorted Source Nodes: [x_216, x_217, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg46_1
        del buf19
        buf21 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_219, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf21, buf20, arg47_1, arg48_1, arg49_1, arg50_1, 602112, grid=grid(602112), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        del buf20
        # Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg51_1
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_222, x_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf23, arg52_1, arg53_1, arg54_1, arg55_1, 602112, grid=grid(602112), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        # Topologically Sorted Source Nodes: [x_222, x_223, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg56_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf24, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg56_1
        del buf23
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_225, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf25, arg57_1, arg58_1, arg59_1, arg60_1, 602112, grid=grid(602112), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        # Topologically Sorted Source Nodes: [x_225, x_226, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg61_1
        del buf25
        buf27 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_228, x_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf27, buf26, arg62_1, arg63_1, arg64_1, arg65_1, 602112, grid=grid(602112), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        del buf26
        # Topologically Sorted Source Nodes: [x_228, x_229, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 144, 56, 56), (451584, 1, 8064, 144))
        del arg66_1
        del buf27
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_231, x_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf29, arg67_1, arg68_1, arg69_1, arg70_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        # Topologically Sorted Source Nodes: [x_231, x_232, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg71_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf30, (8, 144, 28, 28), (112896, 1, 4032, 144))
        del arg71_1
        del buf29
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf31, arg72_1, arg73_1, arg74_1, arg75_1, 903168, grid=grid(903168), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        # Topologically Sorted Source Nodes: [x_234, x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 32, 28, 28), (25088, 1, 896, 32))
        del arg76_1
        del buf31
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_237], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf33, arg77_1, arg78_1, arg79_1, arg80_1, 200704, grid=grid(200704), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        # Topologically Sorted Source Nodes: [x_238], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 96, 28, 28), (75264, 1, 2688, 96))
        del arg81_1
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_239, x_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf35, arg82_1, arg83_1, arg84_1, arg85_1, 602112, grid=grid(602112), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        # Topologically Sorted Source Nodes: [x_239, x_240, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg86_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf36, (8, 96, 28, 28), (75264, 1, 2688, 96))
        del arg86_1
        del buf35
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_242, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf37, arg87_1, arg88_1, arg89_1, arg90_1, 602112, grid=grid(602112), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        # Topologically Sorted Source Nodes: [x_242, x_243, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 32, 28, 28), (25088, 1, 896, 32))
        del arg91_1
        del buf37
        buf39 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_245, x_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf39, buf38, arg92_1, arg93_1, arg94_1, arg95_1, 200704, grid=grid(200704), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf38
        # Topologically Sorted Source Nodes: [x_247], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg96_1
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf41, arg97_1, arg98_1, arg99_1, arg100_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        # Topologically Sorted Source Nodes: [x_248, x_249, x_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg101_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf42, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg101_1
        del buf41
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_251, x_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf43, arg102_1, arg103_1, arg104_1, arg105_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        # Topologically Sorted Source Nodes: [x_251, x_252, x_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 32, 28, 28), (25088, 1, 896, 32))
        del arg106_1
        del buf43
        buf45 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_254, x_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf45, buf44, arg107_1, arg108_1, arg109_1, arg110_1, 200704, grid=grid(200704), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        del buf44
        # Topologically Sorted Source Nodes: [x_256], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg111_1
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_257, x_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf47, arg112_1, arg113_1, arg114_1, arg115_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        # Topologically Sorted Source Nodes: [x_257, x_258, x_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg116_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf48, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg116_1
        del buf47
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_260, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf49, arg117_1, arg118_1, arg119_1, arg120_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        # Topologically Sorted Source Nodes: [x_260, x_261, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 32, 28, 28), (25088, 1, 896, 32))
        del arg121_1
        del buf49
        buf51 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [x_263, x_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_13.run(buf51, buf50, arg122_1, arg123_1, arg124_1, arg125_1, 200704, grid=grid(200704), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        del buf50
        # Topologically Sorted Source Nodes: [x_263, x_264, x_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg126_1
        del buf51
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_266, x_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf53, arg127_1, arg128_1, arg129_1, arg130_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        # Topologically Sorted Source Nodes: [x_266, x_267, x_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg131_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf54, (8, 192, 14, 14), (37632, 1, 2688, 192))
        del arg131_1
        del buf53
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_269, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf55, arg132_1, arg133_1, arg134_1, arg135_1, 301056, grid=grid(301056), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        # Topologically Sorted Source Nodes: [x_269, x_270, x_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 64, 14, 14), (12544, 1, 896, 64))
        del arg136_1
        del buf55
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_272], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf57, arg137_1, arg138_1, arg139_1, arg140_1, 100352, grid=grid(100352), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        # Topologically Sorted Source Nodes: [x_273], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 192, 14, 14), (37632, 1, 2688, 192))
        del arg141_1
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_274, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf59, arg142_1, arg143_1, arg144_1, arg145_1, 301056, grid=grid(301056), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        # Topologically Sorted Source Nodes: [x_274, x_275, x_276], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg146_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf60, (8, 192, 14, 14), (37632, 1, 2688, 192))
        del arg146_1
        del buf59
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_277, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf61, arg147_1, arg148_1, arg149_1, arg150_1, 301056, grid=grid(301056), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        # Topologically Sorted Source Nodes: [x_277, x_278, x_279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 64, 14, 14), (12544, 1, 896, 64))
        del arg151_1
        del buf61
        buf63 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_280, x_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf63, buf62, arg152_1, arg153_1, arg154_1, arg155_1, 100352, grid=grid(100352), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        del buf62
        # Topologically Sorted Source Nodes: [x_282], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg156_1
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_283, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf65, arg157_1, arg158_1, arg159_1, arg160_1, 602112, grid=grid(602112), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        # Topologically Sorted Source Nodes: [x_283, x_284, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg161_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf66, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg161_1
        del buf65
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_286, x_287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf67, arg162_1, arg163_1, arg164_1, arg165_1, 602112, grid=grid(602112), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        # Topologically Sorted Source Nodes: [x_286, x_287, x_288], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 64, 14, 14), (12544, 1, 896, 64))
        del arg166_1
        del buf67
        buf69 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [x_289, x_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf69, buf68, arg167_1, arg168_1, arg169_1, arg170_1, 100352, grid=grid(100352), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        del buf68
        # Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg171_1
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf71, arg172_1, arg173_1, arg174_1, arg175_1, 602112, grid=grid(602112), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        # Topologically Sorted Source Nodes: [x_292, x_293, x_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg176_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf72, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg176_1
        del buf71
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_295, x_296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf73, arg177_1, arg178_1, arg179_1, arg180_1, 602112, grid=grid(602112), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        # Topologically Sorted Source Nodes: [x_295, x_296, x_297], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 64, 14, 14), (12544, 1, 896, 64))
        del arg181_1
        del buf73
        buf75 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_298, x_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf75, buf74, arg182_1, arg183_1, arg184_1, arg185_1, 100352, grid=grid(100352), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        del buf74
        # Topologically Sorted Source Nodes: [x_298, x_299, x_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg186_1
        del buf75
        buf77 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_301, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf77, arg187_1, arg188_1, arg189_1, arg190_1, 602112, grid=grid(602112), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        # Topologically Sorted Source Nodes: [x_301, x_302, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg191_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf78, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg191_1
        del buf77
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_304, x_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf79, arg192_1, arg193_1, arg194_1, arg195_1, 602112, grid=grid(602112), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        # Topologically Sorted Source Nodes: [x_304, x_305, x_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg196_1
        del buf79
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf81, arg197_1, arg198_1, arg199_1, arg200_1, 175616, grid=grid(175616), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        # Topologically Sorted Source Nodes: [x_308], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg201_1
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [x_309, x_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf83, arg202_1, arg203_1, arg204_1, arg205_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        # Topologically Sorted Source Nodes: [x_309, x_310, x_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf84 = extern_kernels.convolution(buf83, arg206_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf84, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg206_1
        del buf83
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_312, x_313], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf85, arg207_1, arg208_1, arg209_1, arg210_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        # Topologically Sorted Source Nodes: [x_312, x_313, x_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg211_1
        del buf85
        buf87 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_315, x_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf87, buf86, arg212_1, arg213_1, arg214_1, arg215_1, 175616, grid=grid(175616), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        del buf86
        # Topologically Sorted Source Nodes: [x_317], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg216_1
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_318, x_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf89, arg217_1, arg218_1, arg219_1, arg220_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        # Topologically Sorted Source Nodes: [x_318, x_319, x_320], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg221_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf90, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg221_1
        del buf89
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_321, x_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf91, arg222_1, arg223_1, arg224_1, arg225_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        # Topologically Sorted Source Nodes: [x_321, x_322, x_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg226_1
        del buf91
        buf93 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_324, x_325], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf93, buf92, arg227_1, arg228_1, arg229_1, arg230_1, 175616, grid=grid(175616), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        del buf92
        # Topologically Sorted Source Nodes: [x_326], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg231_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 336, 14, 14), (65856, 1, 4704, 336))
        del arg231_1
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_327, x_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf95, arg232_1, arg233_1, arg234_1, arg235_1, 526848, grid=grid(526848), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        # Topologically Sorted Source Nodes: [x_327, x_328, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg236_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
        assert_size_stride(buf96, (8, 336, 14, 14), (65856, 1, 4704, 336))
        del arg236_1
        del buf95
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_330, x_331], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf97, arg237_1, arg238_1, arg239_1, arg240_1, 526848, grid=grid(526848), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        # Topologically Sorted Source Nodes: [x_330, x_331, x_332], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf98 = extern_kernels.convolution(buf97, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg241_1
        del buf97
        buf99 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_333, x_334], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_21.run(buf99, buf98, arg242_1, arg243_1, arg244_1, arg245_1, 175616, grid=grid(175616), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        del buf98
        # Topologically Sorted Source Nodes: [x_333, x_334, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 672, 14, 14), (131712, 1, 9408, 672))
        del arg246_1
        del buf99
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_336, x_337], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf101, arg247_1, arg248_1, arg249_1, arg250_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        # Topologically Sorted Source Nodes: [x_336, x_337, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg251_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf102, (8, 672, 7, 7), (32928, 1, 4704, 672))
        del arg251_1
        del buf101
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_339, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf103, arg252_1, arg253_1, arg254_1, arg255_1, 263424, grid=grid(263424), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        # Topologically Sorted Source Nodes: [x_339, x_340, x_341], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 184, 7, 7), (9016, 1, 1288, 184))
        del arg256_1
        del buf103
        buf105 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_342], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf105, arg257_1, arg258_1, arg259_1, arg260_1, 72128, grid=grid(72128), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        # Topologically Sorted Source Nodes: [x_343], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg261_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
        del arg261_1
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_344, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf107, arg262_1, arg263_1, arg264_1, arg265_1, 432768, grid=grid(432768), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        # Topologically Sorted Source Nodes: [x_344, x_345, x_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg266_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf108, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
        del arg266_1
        del buf107
        buf109 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_347, x_348], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf109, arg267_1, arg268_1, arg269_1, arg270_1, 432768, grid=grid(432768), stream=stream0)
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        # Topologically Sorted Source Nodes: [x_347, x_348, x_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 184, 7, 7), (9016, 1, 1288, 184))
        del arg271_1
        del buf109
        buf111 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_350, x_351], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf111, buf110, arg272_1, arg273_1, arg274_1, arg275_1, 72128, grid=grid(72128), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        del buf110
        # Topologically Sorted Source Nodes: [x_352], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
        del arg276_1
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_353, x_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf113, arg277_1, arg278_1, arg279_1, arg280_1, 432768, grid=grid(432768), stream=stream0)
        del arg277_1
        del arg278_1
        del arg279_1
        del arg280_1
        # Topologically Sorted Source Nodes: [x_353, x_354, x_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf114 = extern_kernels.convolution(buf113, arg281_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf114, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
        del arg281_1
        del buf113
        buf115 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [x_356, x_357], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf115, arg282_1, arg283_1, arg284_1, arg285_1, 432768, grid=grid(432768), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        # Topologically Sorted Source Nodes: [x_356, x_357, x_358], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf116 = extern_kernels.convolution(buf115, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 184, 7, 7), (9016, 1, 1288, 184))
        del arg286_1
        del buf115
        buf117 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [x_359, x_360], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf117, buf116, arg287_1, arg288_1, arg289_1, arg290_1, 72128, grid=grid(72128), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        del buf116
        # Topologically Sorted Source Nodes: [x_361], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
        del arg291_1
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_362, x_363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf119, arg292_1, arg293_1, arg294_1, arg295_1, 432768, grid=grid(432768), stream=stream0)
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        # Topologically Sorted Source Nodes: [x_362, x_363, x_364], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg296_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf120, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
        del arg296_1
        del buf119
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [x_365, x_366], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf121, arg297_1, arg298_1, arg299_1, arg300_1, 432768, grid=grid(432768), stream=stream0)
        del arg297_1
        del arg298_1
        del arg299_1
        del arg300_1
        # Topologically Sorted Source Nodes: [x_365, x_366, x_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf122 = extern_kernels.convolution(buf121, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 184, 7, 7), (9016, 1, 1288, 184))
        del arg301_1
        del buf121
        buf123 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [x_368, x_369], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf123, buf122, arg302_1, arg303_1, arg304_1, arg305_1, 72128, grid=grid(72128), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        del buf122
        # Topologically Sorted Source Nodes: [x_368, x_369, x_370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
        del arg306_1
        del buf123
        buf125 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_371, x_372], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf125, arg307_1, arg308_1, arg309_1, arg310_1, 432768, grid=grid(432768), stream=stream0)
        del arg307_1
        del arg308_1
        del arg309_1
        del arg310_1
        # Topologically Sorted Source Nodes: [x_371, x_372, x_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf126 = extern_kernels.convolution(buf125, arg311_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf126, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
        del arg311_1
        del buf125
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_374, x_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf127, arg312_1, arg313_1, arg314_1, arg315_1, 432768, grid=grid(432768), stream=stream0)
        del arg312_1
        del arg313_1
        del arg314_1
        del arg315_1
        # Topologically Sorted Source Nodes: [x_374, x_375, x_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 352, 7, 7), (17248, 1, 2464, 352))
        del arg316_1
        del buf127
        buf129 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf129, arg317_1, arg318_1, arg319_1, arg320_1, 137984, grid=grid(137984), stream=stream0)
        del arg317_1
        del arg318_1
        del arg319_1
        del arg320_1
        # Topologically Sorted Source Nodes: [x_377, x_378], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf130 = extern_kernels.convolution(buf129, arg321_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 1984, 7, 7), (97216, 1, 13888, 1984))
        del arg321_1
        del buf129
        buf132 = empty_strided_cuda((8, 1984, 1, 1), (1984, 1, 15872, 15872), torch.float32)
        # Topologically Sorted Source Nodes: [x_379, x_380, x_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_28.run(buf130, arg322_1, arg323_1, arg324_1, arg325_1, buf132, 15872, 49, grid=grid(15872), stream=stream0)
        del arg322_1
        del arg323_1
        del arg324_1
        del arg325_1
        del buf130
        buf133 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_383], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg327_1, reinterpret_tensor(buf132, (8, 1984), (1984, 1), 0), reinterpret_tensor(arg326_1, (1984, 1000), (1, 1984), 0), alpha=1, beta=1, out=buf133)
        del arg326_1
        del arg327_1
        del buf132
    return (buf133, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((32, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((96, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((96, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((32, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((192, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((384, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((384, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((112, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    arg211_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((336, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((112, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((184, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((184, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((184, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((184, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1104, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((352, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1984, 352, 1, 1), (352, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1000, 1984), (1984, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('fbnetc_100', benchmark_compiled_module)
