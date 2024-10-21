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


# kernel path: /tmp/torchinductor_sahanp/pk/cpkhbgeg2kcqjjmkgqixokomy6bkct57mjcapnwcdeogg2c252d7.py
# Topologically Sorted Source Nodes: [x_203], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_203 => constant_pad_nd_15
# Graph fragment:
#   %constant_pad_nd_15 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg1_1, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_0 = async_compile.triton('triton_poi_fused_constant_pad_nd_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 50625
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 225)
    x2 = xindex % 225
    y4 = yindex
    x5 = xindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = x3
    tmp1 = tl.full([1, 1], 224, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (224*x3) + (50176*y4)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (y0 + (3*x5) + (151875*y1)), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jf/cjfodmtfuxyzhxsvkyg4ks5pht5fymfjilyro6kqrtfpo5pydiqi.py
# Topologically Sorted Source Nodes: [x_203, x_204], Original ATen: [aten.constant_pad_nd, aten.convolution]
# Source node to ATen node mapping:
#   x_203 => constant_pad_nd_15
#   x_204 => convolution_155
# Graph fragment:
#   %constant_pad_nd_15 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%arg1_1, [0, 1, 0, 1], 0.0), kwargs = {})
#   %convolution_155 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_15, %arg0_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_constant_pad_nd_convolution_1 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
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


# kernel path: /tmp/torchinductor_sahanp/dz/cdz2rgwmfgf5b2ceee56bwfxzjzmhv5omefqcoxgdvcypfsprn27.py
# Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_205 => add_131, mul_239, mul_240, sub_58
#   x_206 => relu_7
# Graph fragment:
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_155, %unsqueeze_465), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_467), kwargs = {})
#   %mul_240 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_239, %unsqueeze_469), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_240, %unsqueeze_471), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_131,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/y7/cy7ztr2bzlhdvvgj6v7cdxlvw5vaaw4uqep44cz6vx4ymgqcjxud.py
# Topologically Sorted Source Nodes: [x_211, x_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_211 => add_135, mul_245, mul_246, sub_60
#   x_212 => add_136
# Graph fragment:
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_157, %unsqueeze_481), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_483), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_245, %unsqueeze_485), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_246, %unsqueeze_487), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_135, %relu_7), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 32
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 12544
    y1 = (yindex // 12544)
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (y0 + (12544*x2) + (401408*y1)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n6/cn6y5zxcpxzgfsldauiff7ei6t2qhqals7axr57tkvutnmrblfcd.py
# Topologically Sorted Source Nodes: [conv2d_158], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_158 => convolution_158
# Graph fragment:
#   %convolution_158 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_320, %arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_4 = async_compile.triton('triton_poi_fused_convolution_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[128, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y0) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sr/csrovwkgpxqtyjduwc4iytgxv7jjp4na67dgwflarrftibv5g4bt.py
# Topologically Sorted Source Nodes: [conv2d_159], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_159 => convolution_159
# Graph fragment:
#   %convolution_159 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_321, %arg17_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_5 = async_compile.triton('triton_poi_fused_convolution_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[128, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (200704 + x2 + (12544*y0) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uc/cucsslhyjreh3nlzxm6vjm4dcnsqd5ainw56hbgjn2bqh64tqmij.py
# Topologically Sorted Source Nodes: [x_213, x_214, x_215], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_213 => cat_41
#   x_214 => add_138, mul_248, mul_249, sub_61
#   x_215 => relu_9
# Graph fragment:
#   %cat_41 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_158, %convolution_159], 1), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_41, %unsqueeze_489), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_248, %unsqueeze_493), kwargs = {})
#   %add_138 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_249, %unsqueeze_495), kwargs = {})
#   %relu_9 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_138,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 12544) % 192
    x0 = xindex % 12544
    x2 = (xindex // 2408448)
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((96*x0) + (1204224*x2) + x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((96*x0) + (1204224*x2) + ((-96) + x1)), tmp6, eviction_policy='evict_last', other=0.0)
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
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/54/c545sgiroz37z57ozved5dbpqz2zh5cteazcohuhvjz5442sbjf4.py
# Topologically Sorted Source Nodes: [x_216], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_216 => constant_pad_nd_16
# Graph fragment:
#   %constant_pad_nd_16 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_325, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_7 = async_compile.triton('triton_poi_fused_constant_pad_nd_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 12769
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 113)
    x2 = xindex % 113
    y0 = yindex % 64
    y1 = (yindex // 64)
    x4 = xindex
    tmp0 = x3
    tmp1 = tl.full([1, 1], 112, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (112*x3) + (12544*y0) + (2408448*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (y0 + (64*x4) + (817216*y1)), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tq/ctqgojyug2vtvabmueqlebwziqrm4nhztcxhluligb32axda6ptq.py
# Topologically Sorted Source Nodes: [x_217], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_217 => constant_pad_nd_17
# Graph fragment:
#   %constant_pad_nd_17 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_329, [1, 2, 1, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_8 = async_compile.triton('triton_poi_fused_constant_pad_nd_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 13225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 115)
    x2 = xindex % 115
    y0 = yindex % 64
    y1 = (yindex // 64)
    x5 = xindex
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (802703 + x2 + (112*x3) + (12544*y0) + (2408448*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (y0 + (64*x5) + (846400*y1)), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wx/cwxo3fx6ylqcsbhe4j2wn646u57ou77ffdoqve5otaue3jir72i5.py
# Topologically Sorted Source Nodes: [x_218], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_218 => constant_pad_nd_18
# Graph fragment:
#   %constant_pad_nd_18 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_333, [2, 3, 2, 3], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_9 = async_compile.triton('triton_poi_fused_constant_pad_nd_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 13689
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 117)
    x2 = xindex % 117
    y0 = yindex % 64
    y1 = (yindex // 64)
    x5 = xindex
    tmp0 = (-2) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (1605406 + x2 + (112*x3) + (12544*y0) + (2408448*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (y0 + (64*x5) + (876096*y1)), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/j5/cj5tbnc7a5xtzpaptatvhxaj67vtxsbtetntzsshnqwudy5dco6f.py
# Topologically Sorted Source Nodes: [x_219, x_220, x_221], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_219 => cat_42
#   x_220 => add_140, mul_251, mul_252, sub_62
#   x_221 => relu_10
# Graph fragment:
#   %cat_42 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_160, %convolution_161, %convolution_162], 1), kwargs = {})
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_42, %unsqueeze_497), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %unsqueeze_499), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_251, %unsqueeze_501), kwargs = {})
#   %add_140 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_252, %unsqueeze_503), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_140,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 3136) % 192
    x0 = xindex % 3136
    x2 = (xindex // 602112)
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((64*x0) + (200704*x2) + x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((64*x0) + (200704*x2) + ((-64) + x1)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + ((64*x0) + (200704*x2) + ((-128) + x1)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 0.001
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.sqrt(tmp21)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = 1.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1], 0, tl.int32)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tl.store(out_ptr1 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gq/cgqhukl5axi4utir2wrqwktquqbntfjzou2s4btutspwwo4oysoo.py
# Topologically Sorted Source Nodes: [conv2d_163], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_163 => convolution_163
# Graph fragment:
#   %convolution_163 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_336, %arg29_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_11 = async_compile.triton('triton_poi_fused_convolution_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y0) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2j/c2j5vjdhbtteorupwp3c3rb3ei3d3ncord6bwbj7tlrdm6mvsq4q.py
# Topologically Sorted Source Nodes: [conv2d_164], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_164 => convolution_164
# Graph fragment:
#   %convolution_164 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_339, %arg30_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_12 = async_compile.triton('triton_poi_fused_convolution_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (301056 + x2 + (3136*y0) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5x/c5xejbmp5tx7bpg2ukpmgcbf572v5l4mjndqbymrnnkq2hiqgbe2.py
# Topologically Sorted Source Nodes: [x_222, x_223], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_222 => cat_43
#   x_223 => add_142, mul_254, mul_255, sub_63
# Graph fragment:
#   %cat_43 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_163, %convolution_164], 1), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_43, %unsqueeze_505), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_255 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_254, %unsqueeze_509), kwargs = {})
#   %add_142 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_255, %unsqueeze_511), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 3136) % 40
    x0 = xindex % 3136
    x2 = (xindex // 125440)
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((20*x0) + (62720*x2) + x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 40, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((20*x0) + (62720*x2) + ((-20) + x1)), tmp6, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2h/c2h4lccwro64fbi3ecmyxed5fbo6nyi6f5aff6jl6x2vb3k4iz2b.py
# Topologically Sorted Source Nodes: [conv2d_165], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_165 => convolution_165
# Graph fragment:
#   %convolution_165 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_340, %arg35_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_14 = async_compile.triton('triton_poi_fused_convolution_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 160
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 20
    y1 = (yindex // 20)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y0) + (125440*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (20*x2) + (62720*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ua/cuaup2ek4asckotgczgnlkeoy2isjts6qzoj4zjpep3kg5wy72c5.py
# Topologically Sorted Source Nodes: [conv2d_166], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_166 => convolution_166
# Graph fragment:
#   %convolution_166 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_341, %arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_15 = async_compile.triton('triton_poi_fused_convolution_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 160
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 20
    y1 = (yindex // 20)
    tmp0 = tl.load(in_ptr0 + (62720 + x2 + (3136*y0) + (125440*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (20*x2) + (62720*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/67/c674dtws36sc6aj3mim5ir4ggmvesapja4modn3rzlfucteisnvz.py
# Topologically Sorted Source Nodes: [x_224, x_225, x_226], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_224 => cat_44
#   x_225 => add_144, mul_257, mul_258, sub_64
#   x_226 => relu_11
# Graph fragment:
#   %cat_44 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_165, %convolution_166], 1), kwargs = {})
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_44, %unsqueeze_513), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_515), kwargs = {})
#   %mul_258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_257, %unsqueeze_517), kwargs = {})
#   %add_144 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_258, %unsqueeze_519), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_144,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 120
    x1 = (xindex // 120)
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((60*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 120, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((60*x1) + ((-60) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
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
    tmp26 = tl.full([1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/v5/cv5kltn6is52hmnom33c5mt2o3mxj72zvne6ar6a7mucwdzpo6sl.py
# Topologically Sorted Source Nodes: [x_228, x_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_228 => add_146, mul_260, mul_261, sub_65
#   x_229 => relu_12
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_167, %unsqueeze_521), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_261 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_260, %unsqueeze_525), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_261, %unsqueeze_527), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_146,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 120
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


# kernel path: /tmp/torchinductor_sahanp/sh/cshs5l2l5ic7asfgvs7xzy4inbuc7k37lg4d5dla7l7o4many2ru.py
# Topologically Sorted Source Nodes: [x_230, x_231, x_232], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_230 => cat_45
#   x_231 => add_148, mul_263, mul_264, sub_66
#   x_232 => add_149
# Graph fragment:
#   %cat_45 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_168, %convolution_169], 1), kwargs = {})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_45, %unsqueeze_529), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_264 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_263, %unsqueeze_533), kwargs = {})
#   %add_148 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_264, %unsqueeze_535), kwargs = {})
#   %add_149 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_148, %add_142), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 40
    x2 = xindex
    y1 = (yindex // 40)
    y3 = yindex
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((20*x2) + (62720*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 40, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((20*x2) + (62720*y1) + ((-20) + y0)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr0 + (y0 + (40*x2) + (125440*y1)), tmp27, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ck/ccke63krazdmm6qtce5oimaprgwq3h3kefirhjpe6q5odu5hn7wh.py
# Topologically Sorted Source Nodes: [x_234], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_234 => add_151, mul_266, mul_267, sub_67
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_170, %unsqueeze_537), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_266, %unsqueeze_541), kwargs = {})
#   %add_151 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_267, %unsqueeze_543), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 240
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


# kernel path: /tmp/torchinductor_sahanp/rz/crz3o35qnf2d4qcej25uvb4ifpncvrzbde2sb3jauaspusz6wscw.py
# Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_236 => constant_pad_nd_19
# Graph fragment:
#   %constant_pad_nd_19 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_352, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_20 = async_compile.triton('triton_poi_fused_constant_pad_nd_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1559520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3420) % 57
    x1 = (xindex // 60) % 57
    x0 = xindex % 60
    x3 = (xindex // 194940)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 56, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (240*x1) + (13440*x2) + (752640*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xv/cxvysus4oznepidaof4lsuvmudmxdh7hjzevepphvwrmgwoslb6n.py
# Topologically Sorted Source Nodes: [x_237], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_237 => constant_pad_nd_20
# Graph fragment:
#   %constant_pad_nd_20 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_357, [1, 2, 1, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_21 = async_compile.triton('triton_poi_fused_constant_pad_nd_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1670880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3540) % 59
    x1 = (xindex // 60) % 59
    x0 = xindex % 60
    x3 = (xindex // 208860)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-13620) + x0 + (240*x1) + (13440*x2) + (752640*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fx/cfx3nbichy27xrrywgfbqhcqunxmyjvwel67n4lqwbwi2rbt4oyq.py
# Topologically Sorted Source Nodes: [x_238], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_238 => constant_pad_nd_21
# Graph fragment:
#   %constant_pad_nd_21 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_362, [2, 3, 2, 3], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_22 = async_compile.triton('triton_poi_fused_constant_pad_nd_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1786080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3660) % 61
    x1 = (xindex // 60) % 61
    x0 = xindex % 60
    x3 = (xindex // 223260)
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-27240) + x0 + (240*x1) + (13440*x2) + (752640*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/s3/cs3mhsugsgurs37sbxtm2l6txfh2jexlwsksftv7lwx2w5etkfuq.py
# Topologically Sorted Source Nodes: [x_239], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_239 => constant_pad_nd_22
# Graph fragment:
#   %constant_pad_nd_22 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_367, [3, 4, 3, 4], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_23 = async_compile.triton('triton_poi_fused_constant_pad_nd_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1905120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3780) % 63
    x1 = (xindex // 60) % 63
    x0 = xindex % 60
    x3 = (xindex // 238140)
    x6 = xindex
    tmp0 = (-3) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-40860) + x0 + (240*x1) + (13440*x2) + (752640*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dw/cdwvzbo3nbbni2kawske462wjtns6glx7wplrwfdehfccubu4nao.py
# Topologically Sorted Source Nodes: [x_240, x_241, x_242, x_se_64], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_240 => cat_46
#   x_241 => add_153, mul_270, mul_271, sub_68
#   x_242 => mul_272, sigmoid_65
#   x_se_64 => mean_17
# Graph fragment:
#   %cat_46 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_171, %convolution_172, %convolution_173, %convolution_174], 1), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_46, %unsqueeze_545), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_270, %unsqueeze_549), kwargs = {})
#   %add_153 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_271, %unsqueeze_551), kwargs = {})
#   %sigmoid_65 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_153,), kwargs = {})
#   %mul_272 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_153, %sigmoid_65), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_272, [2, 3], True), kwargs = {})
triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_24 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    x3 = xindex
    tmp34 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 60, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((60*r2) + (47040*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 120, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tmp6 & tmp8
        tmp10 = tl.load(in_ptr1 + ((60*r2) + (47040*x1) + ((-60) + x0)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp0 >= tmp7
        tmp12 = tl.full([1, 1], 180, tl.int64)
        tmp13 = tmp0 < tmp12
        tmp14 = tmp11 & tmp13
        tmp15 = tl.load(in_ptr2 + ((60*r2) + (47040*x1) + ((-120) + x0)), rmask & tmp14 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp0 >= tmp12
        tmp17 = tl.full([1, 1], 240, tl.int64)
        tmp18 = tmp0 < tmp17
        tmp19 = tl.load(in_ptr3 + ((60*r2) + (47040*x1) + ((-180) + x0)), rmask & tmp16 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.where(tmp14, tmp15, tmp19)
        tmp21 = tl.where(tmp9, tmp10, tmp20)
        tmp22 = tl.where(tmp4, tmp5, tmp21)
        tmp24 = tmp22 - tmp23
        tmp26 = 0.001
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.sqrt(tmp27)
        tmp29 = tl.full([1, 1], 1, tl.int32)
        tmp30 = tmp29 / tmp28
        tmp31 = 1.0
        tmp32 = tmp30 * tmp31
        tmp33 = tmp24 * tmp32
        tmp35 = tmp33 * tmp34
        tmp37 = tmp35 + tmp36
        tmp38 = tl.sigmoid(tmp37)
        tmp39 = tmp37 * tmp38
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask & xmask, tmp42, _tmp41)
        tl.store(out_ptr0 + (r2 + (784*x3)), tmp33, rmask & xmask)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tmp43 = 784.0
    tmp44 = tmp41 / tmp43
    tl.store(out_ptr2 + (x3), tmp44, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/i4/ci4v6fjg3cz7xxxczevdicrp6vgjq3o6tbak6vzalmpwgavbi7ki.py
# Topologically Sorted Source Nodes: [x_241, x_242, x_se_64, x_se_65, x_se_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_241 => add_153, mul_271
#   x_242 => mul_272, sigmoid_65
#   x_se_64 => mean_17
#   x_se_65 => convolution_175
#   x_se_66 => mul_273, sigmoid_66
# Graph fragment:
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_270, %unsqueeze_549), kwargs = {})
#   %add_153 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_271, %unsqueeze_551), kwargs = {})
#   %sigmoid_65 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_153,), kwargs = {})
#   %mul_272 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_153, %sigmoid_65), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_272, [2, 3], True), kwargs = {})
#   %convolution_175 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_17, %arg65_1, %arg66_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_66 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_175,), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_175, %sigmoid_66), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_25(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 20
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3l/c3l5bgzffxmg3vgvuqizkm3hkkk6sdsodwx3n4uzsh7355tn2g6b.py
# Topologically Sorted Source Nodes: [x_241, x_242, x_se_64, x_se_65, x_se_66, x_se_67, sigmoid_16, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_16 => sigmoid_67
#   x_241 => add_153, mul_271
#   x_242 => mul_272, sigmoid_65
#   x_243 => mul_274
#   x_se_64 => mean_17
#   x_se_65 => convolution_175
#   x_se_66 => mul_273, sigmoid_66
#   x_se_67 => convolution_176
# Graph fragment:
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_270, %unsqueeze_549), kwargs = {})
#   %add_153 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_271, %unsqueeze_551), kwargs = {})
#   %sigmoid_65 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_153,), kwargs = {})
#   %mul_272 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_153, %sigmoid_65), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_272, [2, 3], True), kwargs = {})
#   %convolution_175 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_17, %arg65_1, %arg66_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_66 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_175,), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_175, %sigmoid_66), kwargs = {})
#   %convolution_176 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_273, %arg67_1, %arg68_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_67 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_176,), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_272, %sigmoid_67), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp6 * tmp10
    tl.store(out_ptr0 + (y0 + (240*x2) + (188160*y1)), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ml/cmllmmnd635szocog5obhzcea57wg3wrplm6n2nvhwyx45fqnu73.py
# Topologically Sorted Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_245 => add_155, mul_276, mul_277, sub_69
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_177, %unsqueeze_553), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_276, %unsqueeze_557), kwargs = {})
#   %add_155 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_277, %unsqueeze_559), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 56
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


# kernel path: /tmp/torchinductor_sahanp/dn/cdn3fxikfl6kxy5v3e4tu4g4n6o6pfg2t6qjj2jpchsm7osg3bgz.py
# Topologically Sorted Source Nodes: [x_246, x_247, x_248], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_246 => cat_47
#   x_247 => add_157, mul_279, mul_280, sub_70
#   x_248 => mul_281, sigmoid_68
# Graph fragment:
#   %cat_47 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_178, %convolution_179], 1), kwargs = {})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_47, %unsqueeze_561), kwargs = {})
#   %mul_279 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_563), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_279, %unsqueeze_565), kwargs = {})
#   %add_157 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_280, %unsqueeze_567), kwargs = {})
#   %sigmoid_68 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_157,), kwargs = {})
#   %mul_281 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_157, %sigmoid_68), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 336
    x0 = xindex % 784
    x2 = (xindex // 263424)
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 168, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((168*x0) + (131712*x2) + x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 336, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((168*x0) + (131712*x2) + ((-168) + x1)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp26 = tl.sigmoid(tmp25)
    tmp27 = tmp25 * tmp26
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mq/cmqcckilyiww4g5stfoukvujf7dde4cinzznvz6bei4yeyllqryh.py
# Topologically Sorted Source Nodes: [conv2d_180], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_180 => convolution_180
# Graph fragment:
#   %convolution_180 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_372, %arg80_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168), kwargs = {})
triton_poi_fused_convolution_29 = async_compile.triton('triton_poi_fused_convolution_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1344
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 168
    y1 = (yindex // 168)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y0) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (168*x2) + (131712*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rp/crp2npxxdutgu5ogdff7c26j3krxzbnln4ezmzsdzsjbsoaq5boo.py
# Topologically Sorted Source Nodes: [conv2d_181], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_181 => convolution_181
# Graph fragment:
#   %convolution_181 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_375, %arg81_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168), kwargs = {})
triton_poi_fused_convolution_30 = async_compile.triton('triton_poi_fused_convolution_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1344
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 168
    y1 = (yindex // 168)
    tmp0 = tl.load(in_ptr0 + (131712 + x2 + (784*y0) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (168*x2) + (131712*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zm/czmj7q37tjamt2fwfpi5yk2eqz5i5h3nmsuup7hcpbvypgmlt5nm.py
# Topologically Sorted Source Nodes: [x_249, x_250, x_251, x_se_68], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_249 => cat_48
#   x_250 => add_159, mul_283, mul_284, sub_71
#   x_251 => mul_285, sigmoid_69
#   x_se_68 => mean_18
# Graph fragment:
#   %cat_48 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_180, %convolution_181], 1), kwargs = {})
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_48, %unsqueeze_569), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_571), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_283, %unsqueeze_573), kwargs = {})
#   %add_159 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_284, %unsqueeze_575), kwargs = {})
#   %sigmoid_69 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_159,), kwargs = {})
#   %mul_285 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_159, %sigmoid_69), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_285, [2, 3], True), kwargs = {})
triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_31 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 336
    x1 = (xindex // 336)
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    x3 = xindex
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 168, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((168*r2) + (131712*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 336, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((168*r2) + (131712*x1) + ((-168) + x0)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp26 = tl.sigmoid(tmp25)
        tmp27 = tmp25 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
        tl.store(out_ptr0 + (r2 + (784*x3)), tmp25, rmask & xmask)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tmp31 = 784.0
    tmp32 = tmp29 / tmp31
    tl.store(out_ptr2 + (x3), tmp32, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kz/ckzttcwhhvgugvrrwv3gmpyic7yrxmzncu7hlhf6byojz2hkqgo7.py
# Topologically Sorted Source Nodes: [x_251, x_se_68, x_se_69, x_se_70], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_251 => mul_285, sigmoid_69
#   x_se_68 => mean_18
#   x_se_69 => convolution_182
#   x_se_70 => mul_286, sigmoid_70
# Graph fragment:
#   %sigmoid_69 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_159,), kwargs = {})
#   %mul_285 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_159, %sigmoid_69), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_285, [2, 3], True), kwargs = {})
#   %convolution_182 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_18, %arg86_1, %arg87_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_70 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_182,), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_182, %sigmoid_70), kwargs = {})
triton_poi_fused_convolution_mean_silu_32 = async_compile.triton('triton_poi_fused_convolution_mean_silu_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_32(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 28
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d5/cd5sf32a7xiimpmnws2xc5g2kys2omfdp3vbaj42g7kb2aaknlyq.py
# Topologically Sorted Source Nodes: [x_251, x_se_68, x_se_69, x_se_70, x_se_71, sigmoid_17, x_252], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_17 => sigmoid_71
#   x_251 => mul_285, sigmoid_69
#   x_252 => mul_287
#   x_se_68 => mean_18
#   x_se_69 => convolution_182
#   x_se_70 => mul_286, sigmoid_70
#   x_se_71 => convolution_183
# Graph fragment:
#   %sigmoid_69 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_159,), kwargs = {})
#   %mul_285 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_159, %sigmoid_69), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_285, [2, 3], True), kwargs = {})
#   %convolution_182 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_18, %arg86_1, %arg87_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_70 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_182,), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_182, %sigmoid_70), kwargs = {})
#   %convolution_183 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_286, %arg88_1, %arg89_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_71 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_183,), kwargs = {})
#   %mul_287 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_285, %sigmoid_71), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_33 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_33(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 336
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yn/cynuzo4mfzkeucjwpplzoqn75c7jzrciee73bkj6fimw4i7bd5py.py
# Topologically Sorted Source Nodes: [x_253, x_254, x_255], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_253 => cat_49
#   x_254 => add_161, mul_289, mul_290, sub_72
#   x_255 => add_162
# Graph fragment:
#   %cat_49 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_184, %convolution_185], 1), kwargs = {})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_49, %unsqueeze_577), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_581), kwargs = {})
#   %add_161 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %unsqueeze_583), kwargs = {})
#   %add_162 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_161, %add_155), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 56
    x1 = (xindex // 56)
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((28*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 56, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((28*x1) + ((-28) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp27 = tmp25 + tmp26
    tl.store(in_out_ptr0 + (x2), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tv/ctvg4cpanuasrburhoc6e6m6tk64zzftcdkhi566442qsnlui7ov.py
# Topologically Sorted Source Nodes: [x_277], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_277 => add_178, mul_318, mul_319, sub_79
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_202, %unsqueeze_633), kwargs = {})
#   %mul_318 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_635), kwargs = {})
#   %mul_319 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_318, %unsqueeze_637), kwargs = {})
#   %add_178 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_319, %unsqueeze_639), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ml/cmlyjasy2klrvt43gog4jd7tfcrcmvzpba7hlozbahvpz5icfwc3.py
# Topologically Sorted Source Nodes: [x_279], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_279 => constant_pad_nd_23
# Graph fragment:
#   %constant_pad_nd_23 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_401, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_36 = async_compile.triton('triton_poi_fused_constant_pad_nd_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 753536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3248) % 29
    x1 = (xindex // 112) % 29
    x0 = xindex % 112
    x3 = (xindex // 94192)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 28, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (336*x1) + (9408*x2) + (263424*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/l2/cl2kmtmofpllt2foiecwfflui3j255qtbrlz2wptrohbspzsk5uo.py
# Topologically Sorted Source Nodes: [x_280], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_280 => constant_pad_nd_24
# Graph fragment:
#   %constant_pad_nd_24 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_405, [1, 2, 1, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_37 = async_compile.triton('triton_poi_fused_constant_pad_nd_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_37(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 861056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3472) % 31
    x1 = (xindex // 112) % 31
    x0 = xindex % 112
    x3 = (xindex // 107632)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-9632) + x0 + (336*x1) + (9408*x2) + (263424*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/v4/cv4glrgaxq3dcf32n3eqy7haawqk3ym5ka6r5uc74me7dr3valbc.py
# Topologically Sorted Source Nodes: [x_281], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_281 => constant_pad_nd_25
# Graph fragment:
#   %constant_pad_nd_25 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_409, [2, 3, 2, 3], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_38 = async_compile.triton('triton_poi_fused_constant_pad_nd_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 975744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3696) % 33
    x1 = (xindex // 112) % 33
    x0 = xindex % 112
    x3 = (xindex // 121968)
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-19264) + x0 + (336*x1) + (9408*x2) + (263424*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ry/cry5bquph4h5e3o2ebh5kz2xtzfilqofb4gr33dyfznma2gu5wwa.py
# Topologically Sorted Source Nodes: [x_282, x_283, x_284, x_se_80], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_282 => cat_56
#   x_283 => add_180, mul_322, mul_323, sub_80
#   x_284 => mul_324, sigmoid_81
#   x_se_80 => mean_21
# Graph fragment:
#   %cat_56 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_203, %convolution_204, %convolution_205], 1), kwargs = {})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_56, %unsqueeze_641), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_643), kwargs = {})
#   %mul_323 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_322, %unsqueeze_645), kwargs = {})
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_323, %unsqueeze_647), kwargs = {})
#   %sigmoid_81 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_180,), kwargs = {})
#   %mul_324 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_180, %sigmoid_81), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_324, [2, 3], True), kwargs = {})
triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_39 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 336
    x1 = (xindex // 336)
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    x3 = xindex
    tmp30 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp35 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 112, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((112*r2) + (21952*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 224, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tmp6 & tmp8
        tmp10 = tl.load(in_ptr1 + ((112*r2) + (21952*x1) + ((-112) + x0)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp0 >= tmp7
        tmp12 = tl.full([1, 1], 336, tl.int64)
        tmp13 = tmp0 < tmp12
        tmp14 = tl.load(in_ptr2 + ((112*r2) + (21952*x1) + ((-224) + x0)), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.where(tmp9, tmp10, tmp14)
        tmp16 = tl.where(tmp4, tmp5, tmp15)
        tmp18 = tmp16 - tmp17
        tmp20 = 0.001
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.sqrt(tmp21)
        tmp23 = tl.full([1, 1], 1, tl.int32)
        tmp24 = tmp23 / tmp22
        tmp25 = 1.0
        tmp26 = tmp24 * tmp25
        tmp27 = tmp18 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tmp32 = tl.sigmoid(tmp31)
        tmp33 = tmp31 * tmp32
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(rmask & xmask, tmp36, _tmp35)
        tl.store(out_ptr0 + (r2 + (196*x3)), tmp29, rmask & xmask)
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    tmp37 = 196.0
    tmp38 = tmp35 / tmp37
    tl.store(out_ptr2 + (x3), tmp38, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6z/c6zqf3gc5itlolmgi7owohpkvz7kk6rgzqjmk6qmkpgnj5pfzrz2.py
# Topologically Sorted Source Nodes: [x_283, x_284, x_se_80, x_se_81, x_se_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_283 => add_180
#   x_284 => mul_324, sigmoid_81
#   x_se_80 => mean_21
#   x_se_81 => convolution_206
#   x_se_82 => mul_325, sigmoid_82
# Graph fragment:
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_323, %unsqueeze_647), kwargs = {})
#   %sigmoid_81 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_180,), kwargs = {})
#   %mul_324 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_180, %sigmoid_81), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_324, [2, 3], True), kwargs = {})
#   %convolution_206 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_21, %arg152_1, %arg153_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_206,), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_206, %sigmoid_82), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_40(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 14
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/aw/cawxawhuze7fzuis5c25ps2vt52e6uq2zkl3tresrwl5g6qejvni.py
# Topologically Sorted Source Nodes: [x_283, x_284, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_20 => sigmoid_83
#   x_283 => add_180
#   x_284 => mul_324, sigmoid_81
#   x_285 => mul_326
#   x_se_80 => mean_21
#   x_se_81 => convolution_206
#   x_se_82 => mul_325, sigmoid_82
#   x_se_83 => convolution_207
# Graph fragment:
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_323, %unsqueeze_647), kwargs = {})
#   %sigmoid_81 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_180,), kwargs = {})
#   %mul_324 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_180, %sigmoid_81), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_324, [2, 3], True), kwargs = {})
#   %convolution_206 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_21, %arg152_1, %arg153_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_82 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_206,), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_206, %sigmoid_82), kwargs = {})
#   %convolution_207 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_325, %arg154_1, %arg155_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_83 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_207,), kwargs = {})
#   %mul_326 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_324, %sigmoid_83), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 336
    y1 = (yindex // 336)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp4 * tmp8
    tl.store(out_ptr0 + (y0 + (336*x2) + (65856*y1)), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/af/caf6j5fpsmtgc6zdkakzumcf7jcpvra3lqs35j37ywog3ukvz657.py
# Topologically Sorted Source Nodes: [x_287], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_287 => add_182, mul_328, mul_329, sub_81
# Graph fragment:
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_208, %unsqueeze_649), kwargs = {})
#   %mul_328 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_651), kwargs = {})
#   %mul_329 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_328, %unsqueeze_653), kwargs = {})
#   %add_182 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_329, %unsqueeze_655), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_42', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 104
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


# kernel path: /tmp/torchinductor_sahanp/33/c33zhaisxzo4ehrtkylhepjjq4fq2sadnxpgvyojqwyv5szazw7k.py
# Topologically Sorted Source Nodes: [x_288, x_289, x_290], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_288 => cat_57
#   x_289 => add_184, mul_331, mul_332, sub_82
#   x_290 => mul_333, sigmoid_84
# Graph fragment:
#   %cat_57 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_209, %convolution_210], 1), kwargs = {})
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_57, %unsqueeze_657), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_82, %unsqueeze_659), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_331, %unsqueeze_661), kwargs = {})
#   %add_184 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_332, %unsqueeze_663), kwargs = {})
#   %sigmoid_84 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_184,), kwargs = {})
#   %mul_333 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_184, %sigmoid_84), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 978432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 624
    x0 = xindex % 196
    x2 = (xindex // 122304)
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 312, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((312*x0) + (61152*x2) + x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 624, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((312*x0) + (61152*x2) + ((-312) + x1)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp26 = tl.sigmoid(tmp25)
    tmp27 = tmp25 * tmp26
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xu/cxusfweaggpcdf7x7mtujfxqsbfg657pdp27uu2mylsw4vkk7ckq.py
# Topologically Sorted Source Nodes: [conv2d_211], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_211 => convolution_211
# Graph fragment:
#   %convolution_211 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_416, %arg167_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156), kwargs = {})
triton_poi_fused_convolution_44 = async_compile.triton('triton_poi_fused_convolution_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_44(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1248
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 156
    y1 = (yindex // 156)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3h/c3h5bwzilxqhanoatzevbjptsheign7zg4vvzz5jnmgb52sq2vie.py
# Topologically Sorted Source Nodes: [conv2d_212], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_212 => convolution_212
# Graph fragment:
#   %convolution_212 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_421, %arg168_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156), kwargs = {})
triton_poi_fused_convolution_45 = async_compile.triton('triton_poi_fused_convolution_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_45(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1248
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 156
    y1 = (yindex // 156)
    tmp0 = tl.load(in_ptr0 + (30576 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2k/c2kkevvdobcrgduuee5ffmixkzqwxz4gh4mlkcgg7nshqoma3ijt.py
# Topologically Sorted Source Nodes: [conv2d_213], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_213 => convolution_213
# Graph fragment:
#   %convolution_213 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_426, %arg169_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156), kwargs = {})
triton_poi_fused_convolution_46 = async_compile.triton('triton_poi_fused_convolution_46', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_46(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1248
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 156
    y1 = (yindex // 156)
    tmp0 = tl.load(in_ptr0 + (61152 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/j6/cj6dxat53voxswgnunl566iasf2e5jxl5qaggicjxdjmes7hvcgq.py
# Topologically Sorted Source Nodes: [conv2d_214], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_214 => convolution_214
# Graph fragment:
#   %convolution_214 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_431, %arg170_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156), kwargs = {})
triton_poi_fused_convolution_47 = async_compile.triton('triton_poi_fused_convolution_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_47(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1248
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 156
    y1 = (yindex // 156)
    tmp0 = tl.load(in_ptr0 + (91728 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4o/c4o3u5gu4y4q67nbskqv4yutlg5wvhs6krkeftxhyg6ocs4fjoj3.py
# Topologically Sorted Source Nodes: [x_291, x_292, x_293, x_se_84], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_291 => cat_58
#   x_292 => add_186, mul_335, mul_336, sub_83
#   x_293 => mul_337, sigmoid_85
#   x_se_84 => mean_22
# Graph fragment:
#   %cat_58 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_211, %convolution_212, %convolution_213, %convolution_214], 1), kwargs = {})
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_58, %unsqueeze_665), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_667), kwargs = {})
#   %mul_336 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_335, %unsqueeze_669), kwargs = {})
#   %add_186 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_336, %unsqueeze_671), kwargs = {})
#   %sigmoid_85 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_186,), kwargs = {})
#   %mul_337 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_186, %sigmoid_85), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_337, [2, 3], True), kwargs = {})
triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_48 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 624
    x1 = (xindex // 624)
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    x3 = xindex
    tmp34 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 156, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((156*r2) + (30576*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 312, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tmp6 & tmp8
        tmp10 = tl.load(in_ptr1 + ((156*r2) + (30576*x1) + ((-156) + x0)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp0 >= tmp7
        tmp12 = tl.full([1, 1], 468, tl.int64)
        tmp13 = tmp0 < tmp12
        tmp14 = tmp11 & tmp13
        tmp15 = tl.load(in_ptr2 + ((156*r2) + (30576*x1) + ((-312) + x0)), rmask & tmp14 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp0 >= tmp12
        tmp17 = tl.full([1, 1], 624, tl.int64)
        tmp18 = tmp0 < tmp17
        tmp19 = tl.load(in_ptr3 + ((156*r2) + (30576*x1) + ((-468) + x0)), rmask & tmp16 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.where(tmp14, tmp15, tmp19)
        tmp21 = tl.where(tmp9, tmp10, tmp20)
        tmp22 = tl.where(tmp4, tmp5, tmp21)
        tmp24 = tmp22 - tmp23
        tmp26 = 0.001
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.sqrt(tmp27)
        tmp29 = tl.full([1, 1], 1, tl.int32)
        tmp30 = tmp29 / tmp28
        tmp31 = 1.0
        tmp32 = tmp30 * tmp31
        tmp33 = tmp24 * tmp32
        tmp35 = tmp33 * tmp34
        tmp37 = tmp35 + tmp36
        tmp38 = tl.sigmoid(tmp37)
        tmp39 = tmp37 * tmp38
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask & xmask, tmp42, _tmp41)
        tl.store(out_ptr0 + (r2 + (196*x3)), tmp33, rmask & xmask)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tmp43 = 196.0
    tmp44 = tmp41 / tmp43
    tl.store(out_ptr2 + (x3), tmp44, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fr/cfrcfn6xin4znd3q3iuwhfmgjfxoic5ewme44el7pj2gttxvuas5.py
# Topologically Sorted Source Nodes: [x_292, x_293, x_se_84, x_se_85, x_se_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_292 => add_186, mul_336
#   x_293 => mul_337, sigmoid_85
#   x_se_84 => mean_22
#   x_se_85 => convolution_215
#   x_se_86 => mul_338, sigmoid_86
# Graph fragment:
#   %mul_336 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_335, %unsqueeze_669), kwargs = {})
#   %add_186 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_336, %unsqueeze_671), kwargs = {})
#   %sigmoid_85 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_186,), kwargs = {})
#   %mul_337 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_186, %sigmoid_85), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_337, [2, 3], True), kwargs = {})
#   %convolution_215 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_22, %arg175_1, %arg176_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_86 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_215,), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_215, %sigmoid_86), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_49(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 26
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/32/c32lkydtwlp4cspre5uoutex36qxzma4slcmkjfsgb3q4dng5vrz.py
# Topologically Sorted Source Nodes: [x_292, x_293, x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, x_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_21 => sigmoid_87
#   x_292 => add_186, mul_336
#   x_293 => mul_337, sigmoid_85
#   x_294 => mul_339
#   x_se_84 => mean_22
#   x_se_85 => convolution_215
#   x_se_86 => mul_338, sigmoid_86
#   x_se_87 => convolution_216
# Graph fragment:
#   %mul_336 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_335, %unsqueeze_669), kwargs = {})
#   %add_186 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_336, %unsqueeze_671), kwargs = {})
#   %sigmoid_85 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_186,), kwargs = {})
#   %mul_337 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_186, %sigmoid_85), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_337, [2, 3], True), kwargs = {})
#   %convolution_215 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_22, %arg175_1, %arg176_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_86 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_215,), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_215, %sigmoid_86), kwargs = {})
#   %convolution_216 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_338, %arg177_1, %arg178_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_87 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_216,), kwargs = {})
#   %mul_339 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_337, %sigmoid_87), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 978432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 624
    x4 = (xindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp6 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gv/cgv347cygt32c3m5vbvlc52yscgabbv4b5sjlecajtbkdixbrufr.py
# Topologically Sorted Source Nodes: [conv2d_217], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_217 => convolution_217
# Graph fragment:
#   %convolution_217 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_432, %arg179_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_51 = async_compile.triton('triton_poi_fused_convolution_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_51(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2496
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 312
    y1 = (yindex // 312)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (312*x2) + (61152*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wp/cwp5u6fnqkxcdhqpyt6ukokkqpqyxrwcsvdpj2uyzaepm376dzq5.py
# Topologically Sorted Source Nodes: [conv2d_218], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_218 => convolution_218
# Graph fragment:
#   %convolution_218 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_433, %arg180_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_52 = async_compile.triton('triton_poi_fused_convolution_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_52(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2496
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 312
    y1 = (yindex // 312)
    tmp0 = tl.load(in_ptr0 + (61152 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (312*x2) + (61152*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/34/c34ybmhyphgu7pfdcod7sohwaotkulpjyyoboepj5f4ahw5fxofc.py
# Topologically Sorted Source Nodes: [x_295, x_296, x_297], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_295 => cat_59
#   x_296 => add_188, mul_341, mul_342, sub_84
#   x_297 => add_189
# Graph fragment:
#   %cat_59 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_217, %convolution_218], 1), kwargs = {})
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_59, %unsqueeze_673), kwargs = {})
#   %mul_341 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_675), kwargs = {})
#   %mul_342 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_341, %unsqueeze_677), kwargs = {})
#   %add_188 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_342, %unsqueeze_679), kwargs = {})
#   %add_189 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_188, %add_182), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_53 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_53', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_53(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 104
    x1 = (xindex // 104)
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 52, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((52*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 104, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((52*x1) + ((-52) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp27 = tmp25 + tmp26
    tl.store(in_out_ptr0 + (x2), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2i/c2isv76tilpy2ok7ehhnssz3ewhztdczxvd5aj24rstcq7gowmka.py
# Topologically Sorted Source Nodes: [x_319, x_320], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_319 => add_205, mul_370, mul_371, sub_91
#   x_320 => mul_372, sigmoid_96
# Graph fragment:
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_239, %unsqueeze_729), kwargs = {})
#   %mul_370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %unsqueeze_731), kwargs = {})
#   %mul_371 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_370, %unsqueeze_733), kwargs = {})
#   %add_205 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_371, %unsqueeze_735), kwargs = {})
#   %sigmoid_96 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_205,), kwargs = {})
#   %mul_372 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_205, %sigmoid_96), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_54 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_54', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 978432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 624
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6d/c6dbuerhtgmiodmvcchemzvs2pqti6cnytbfwhfgh7qpnp33sm2j.py
# Topologically Sorted Source Nodes: [x_322], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_322 => add_207, mul_374, mul_375, sub_92
# Graph fragment:
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_240, %unsqueeze_737), kwargs = {})
#   %mul_374 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_375 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_374, %unsqueeze_741), kwargs = {})
#   %add_207 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_375, %unsqueeze_743), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_55(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 978432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 624
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


# kernel path: /tmp/torchinductor_sahanp/s6/cs6v3birdxns43r7zvlyrbx7ytfekgbcabwvnuy23pvb3xzuzr5z.py
# Topologically Sorted Source Nodes: [x_323, x_se_96], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_323 => mul_376, sigmoid_97
#   x_se_96 => mean_25
# Graph fragment:
#   %sigmoid_97 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_207,), kwargs = {})
#   %mul_376 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_207, %sigmoid_97), kwargs = {})
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_376, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_56 = async_compile.triton('triton_red_fused_mean_silu_56', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_56(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 624
    x1 = (xindex // 624)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (624*r2) + (61152*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sq/csqyzkwapqyijtrtg5qexewvzb3fcd2owrwofmebig7syrbcxfdx.py
# Topologically Sorted Source Nodes: [x_323, x_se_96], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_323 => mul_376, sigmoid_97
#   x_se_96 => mean_25
# Graph fragment:
#   %sigmoid_97 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_207,), kwargs = {})
#   %mul_376 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_207, %sigmoid_97), kwargs = {})
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_376, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_57 = async_compile.triton('triton_per_fused_mean_silu_57', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_57(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 624
    x1 = (xindex // 624)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (624*r2) + (1248*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/j2/cj2wyerofcmzoxlra4jz7gk2qgru6qoxvmguwta4s7gk5umlx2hs.py
# Topologically Sorted Source Nodes: [x_323, x_se_96, x_se_97, x_se_98], Original ATen: [aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_323 => mul_376, sigmoid_97
#   x_se_96 => mean_25
#   x_se_97 => convolution_241
#   x_se_98 => mul_377, sigmoid_98
# Graph fragment:
#   %sigmoid_97 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_207,), kwargs = {})
#   %mul_376 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_207, %sigmoid_97), kwargs = {})
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_376, [2, 3], True), kwargs = {})
#   %convolution_241 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_25, %arg243_1, %arg244_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_98 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_241,), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_241, %sigmoid_98), kwargs = {})
triton_poi_fused_convolution_mean_silu_58 = async_compile.triton('triton_poi_fused_convolution_mean_silu_58', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_silu_58(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 52
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6z/c6zwuj4szmlwkk5cbhwgxol4bxezmopg3kh5zeacia6eccq6wdd7.py
# Topologically Sorted Source Nodes: [x_323, x_se_96, x_se_97, x_se_98, x_se_99, sigmoid_24, x_324], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_24 => sigmoid_99
#   x_323 => mul_376, sigmoid_97
#   x_324 => mul_378
#   x_se_96 => mean_25
#   x_se_97 => convolution_241
#   x_se_98 => mul_377, sigmoid_98
#   x_se_99 => convolution_242
# Graph fragment:
#   %sigmoid_97 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_207,), kwargs = {})
#   %mul_376 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_207, %sigmoid_97), kwargs = {})
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_376, [2, 3], True), kwargs = {})
#   %convolution_241 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_25, %arg243_1, %arg244_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_98 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_241,), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_241, %sigmoid_98), kwargs = {})
#   %convolution_242 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_377, %arg245_1, %arg246_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_99 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_242,), kwargs = {})
#   %mul_378 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_376, %sigmoid_99), kwargs = {})
triton_poi_fused_convolution_mean_mul_sigmoid_silu_59 = async_compile.triton('triton_poi_fused_convolution_mean_mul_sigmoid_silu_59', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_sigmoid_silu_59(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 978432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 624
    x2 = (xindex // 122304)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + (624*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2k/c2kgwoowfysve6wonocgjat6lrk6p4yqmo523lrv23cmju27uq4a.py
# Topologically Sorted Source Nodes: [x_326], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_326 => add_209, mul_380, mul_381, sub_93
# Graph fragment:
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_243, %unsqueeze_745), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_747), kwargs = {})
#   %mul_381 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_380, %unsqueeze_749), kwargs = {})
#   %add_209 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_381, %unsqueeze_751), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_60 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_60', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_60', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_60(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
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


# kernel path: /tmp/torchinductor_sahanp/6y/c6yseynyoc4ny5aqjo5th5uavkpbl324b3vxmhnztnmyeflqc46p.py
# Topologically Sorted Source Nodes: [x_327, x_328, x_329], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_327 => cat_66
#   x_328 => add_211, mul_383, mul_384, sub_94
#   x_329 => mul_385, sigmoid_100
# Graph fragment:
#   %cat_66 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_244, %convolution_245], 1), kwargs = {})
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_66, %unsqueeze_753), kwargs = {})
#   %mul_383 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %mul_384 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_383, %unsqueeze_757), kwargs = {})
#   %add_211 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_384, %unsqueeze_759), kwargs = {})
#   %sigmoid_100 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_211,), kwargs = {})
#   %mul_385 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_211, %sigmoid_100), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_61 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_61', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 480
    x0 = xindex % 196
    x2 = (xindex // 94080)
    x3 = xindex
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 240, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((240*x0) + (47040*x2) + x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 480, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((240*x0) + (47040*x2) + ((-240) + x1)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp26 = tl.sigmoid(tmp25)
    tmp27 = tmp25 * tmp26
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3m/c3mmzspcyapkopz4phhoeonk65bm2veeawvsqbq7amzj6e2fuqgy.py
# Topologically Sorted Source Nodes: [conv2d_246], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_246 => convolution_246
# Graph fragment:
#   %convolution_246 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_488, %arg258_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120), kwargs = {})
triton_poi_fused_convolution_62 = async_compile.triton('triton_poi_fused_convolution_62', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_62(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nu/cnuqddjega5dqlubdemrhkmvfidx2ryhm4v3umi5cl3jmt64y53q.py
# Topologically Sorted Source Nodes: [conv2d_247], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_247 => convolution_247
# Graph fragment:
#   %convolution_247 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_493, %arg259_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120), kwargs = {})
triton_poi_fused_convolution_63 = async_compile.triton('triton_poi_fused_convolution_63', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_63(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (23520 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hl/chl2dr62yxxbocgvdvqj5soz7urf22nhojnbvqyn7dysjzlly3ol.py
# Topologically Sorted Source Nodes: [conv2d_248], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_248 => convolution_248
# Graph fragment:
#   %convolution_248 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_498, %arg260_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120), kwargs = {})
triton_poi_fused_convolution_64 = async_compile.triton('triton_poi_fused_convolution_64', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_64(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (47040 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xw/cxwhk4ipoyo66qzson4wwmk5novlvaeuu34s2n2ajkk5kotgayyf.py
# Topologically Sorted Source Nodes: [conv2d_249], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_249 => convolution_249
# Graph fragment:
#   %convolution_249 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_503, %arg261_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120), kwargs = {})
triton_poi_fused_convolution_65 = async_compile.triton('triton_poi_fused_convolution_65', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_65(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (70560 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mk/cmkpcqn7zbrctnlrpsm6r5jz22csx7uzltgfnchgcduf7hwylerh.py
# Topologically Sorted Source Nodes: [x_330, x_331, x_332, x_se_100], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_330 => cat_67
#   x_331 => add_213, mul_387, mul_388, sub_95
#   x_332 => mul_389, sigmoid_101
#   x_se_100 => mean_26
# Graph fragment:
#   %cat_67 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_246, %convolution_247, %convolution_248, %convolution_249], 1), kwargs = {})
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_67, %unsqueeze_761), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %unsqueeze_763), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_387, %unsqueeze_765), kwargs = {})
#   %add_213 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_388, %unsqueeze_767), kwargs = {})
#   %sigmoid_101 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_213,), kwargs = {})
#   %mul_389 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_213, %sigmoid_101), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_389, [2, 3], True), kwargs = {})
triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_66 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_66', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_66(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    x3 = xindex
    tmp34 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 120, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((120*r2) + (23520*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 240, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tmp6 & tmp8
        tmp10 = tl.load(in_ptr1 + ((120*r2) + (23520*x1) + ((-120) + x0)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp0 >= tmp7
        tmp12 = tl.full([1, 1], 360, tl.int64)
        tmp13 = tmp0 < tmp12
        tmp14 = tmp11 & tmp13
        tmp15 = tl.load(in_ptr2 + ((120*r2) + (23520*x1) + ((-240) + x0)), rmask & tmp14 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp0 >= tmp12
        tmp17 = tl.full([1, 1], 480, tl.int64)
        tmp18 = tmp0 < tmp17
        tmp19 = tl.load(in_ptr3 + ((120*r2) + (23520*x1) + ((-360) + x0)), rmask & tmp16 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.where(tmp14, tmp15, tmp19)
        tmp21 = tl.where(tmp9, tmp10, tmp20)
        tmp22 = tl.where(tmp4, tmp5, tmp21)
        tmp24 = tmp22 - tmp23
        tmp26 = 0.001
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.sqrt(tmp27)
        tmp29 = tl.full([1, 1], 1, tl.int32)
        tmp30 = tmp29 / tmp28
        tmp31 = 1.0
        tmp32 = tmp30 * tmp31
        tmp33 = tmp24 * tmp32
        tmp35 = tmp33 * tmp34
        tmp37 = tmp35 + tmp36
        tmp38 = tl.sigmoid(tmp37)
        tmp39 = tmp37 * tmp38
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask & xmask, tmp42, _tmp41)
        tl.store(out_ptr0 + (r2 + (196*x3)), tmp33, rmask & xmask)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tmp43 = 196.0
    tmp44 = tmp41 / tmp43
    tl.store(out_ptr2 + (x3), tmp44, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wr/cwrlyxs6knhseengpu7hbyrptlymvsrpvr6ofuhhgctnwahcawn7.py
# Topologically Sorted Source Nodes: [x_331, x_332, x_se_100, x_se_101, x_se_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_331 => add_213, mul_388
#   x_332 => mul_389, sigmoid_101
#   x_se_100 => mean_26
#   x_se_101 => convolution_250
#   x_se_102 => mul_390, sigmoid_102
# Graph fragment:
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_387, %unsqueeze_765), kwargs = {})
#   %add_213 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_388, %unsqueeze_767), kwargs = {})
#   %sigmoid_101 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_213,), kwargs = {})
#   %mul_389 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_213, %sigmoid_101), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_389, [2, 3], True), kwargs = {})
#   %convolution_250 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_26, %arg266_1, %arg267_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_250,), kwargs = {})
#   %mul_390 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_250, %sigmoid_102), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_67 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_67', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_67', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_67(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gi/cgijv5fjf7ydolee2u6uy5infjmir7wp3hk26wpojrmx3r3rgris.py
# Topologically Sorted Source Nodes: [x_331, x_332, x_se_100, x_se_101, x_se_102, x_se_103, sigmoid_25, x_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_25 => sigmoid_103
#   x_331 => add_213, mul_388
#   x_332 => mul_389, sigmoid_101
#   x_333 => mul_391
#   x_se_100 => mean_26
#   x_se_101 => convolution_250
#   x_se_102 => mul_390, sigmoid_102
#   x_se_103 => convolution_251
# Graph fragment:
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_387, %unsqueeze_765), kwargs = {})
#   %add_213 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_388, %unsqueeze_767), kwargs = {})
#   %sigmoid_101 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_213,), kwargs = {})
#   %mul_389 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_213, %sigmoid_101), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_389, [2, 3], True), kwargs = {})
#   %convolution_250 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_26, %arg266_1, %arg267_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_250,), kwargs = {})
#   %mul_390 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_250, %sigmoid_102), kwargs = {})
#   %convolution_251 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_390, %arg268_1, %arg269_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_103 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_251,), kwargs = {})
#   %mul_391 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_389, %sigmoid_103), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_68 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_68', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_68', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_68(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 480
    x4 = (xindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp6 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/iz/cizqccrm4iqfinpccl24vqzye2hc4e35tgxrdgrf5wsbtpyp6732.py
# Topologically Sorted Source Nodes: [conv2d_252], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_252 => convolution_252
# Graph fragment:
#   %convolution_252 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_504, %arg270_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_69 = async_compile.triton('triton_poi_fused_convolution_69', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_69(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/u2/cu22ed5zr2wezaeastxfd4yr6xkzfwa5mg54v4eilq7nu36pgsth.py
# Topologically Sorted Source Nodes: [conv2d_253], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_253 => convolution_253
# Graph fragment:
#   %convolution_253 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_505, %arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_70 = async_compile.triton('triton_poi_fused_convolution_70', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_70(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (47040 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/k3/ck3mnj5j3fzg2j65foxno63ei7zh4ex6vimh37cwemgkjrwdgbn6.py
# Topologically Sorted Source Nodes: [x_334, x_335, x_336], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_334 => cat_68
#   x_335 => add_215, mul_393, mul_394, sub_96
#   x_336 => add_216
# Graph fragment:
#   %cat_68 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_252, %convolution_253], 1), kwargs = {})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_68, %unsqueeze_769), kwargs = {})
#   %mul_393 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %unsqueeze_771), kwargs = {})
#   %mul_394 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_393, %unsqueeze_773), kwargs = {})
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_394, %unsqueeze_775), kwargs = {})
#   %add_216 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_215, %add_209), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_71 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_71', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_71', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_71(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 160
    x1 = (xindex // 160)
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((80*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((80*x1) + ((-80) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp27 = tmp25 + tmp26
    tl.store(in_out_ptr0 + (x2), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cz/cczpdhjochp34i77jxhzp4apqxw4hmt2cah3eby5qnctckuq2mbo.py
# Topologically Sorted Source Nodes: [x_358], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_358 => add_232, mul_422, mul_423, sub_103
# Graph fragment:
#   %sub_103 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_274, %unsqueeze_825), kwargs = {})
#   %mul_422 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_103, %unsqueeze_827), kwargs = {})
#   %mul_423 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_422, %unsqueeze_829), kwargs = {})
#   %add_232 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_423, %unsqueeze_831), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_72 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_72', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_72', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_72(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
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


# kernel path: /tmp/torchinductor_sahanp/pd/cpd2j6gnguboetkk7x3z3gwkapelw5ndzzi2u42pkptcwn3lm4cv.py
# Topologically Sorted Source Nodes: [x_360], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_360 => constant_pad_nd_26
# Graph fragment:
#   %constant_pad_nd_26 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_558, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_73 = async_compile.triton('triton_poi_fused_constant_pad_nd_73', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_73', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_73(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 432000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3600) % 15
    x1 = (xindex // 240) % 15
    x0 = xindex % 240
    x3 = (xindex // 54000)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 14, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (960*x1) + (13440*x2) + (188160*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5d/c5diu4tjdu5ci27cswwjiujbfcg4nsepqcpjjhjfs5rzdoonfk6b.py
# Topologically Sorted Source Nodes: [x_361], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_361 => constant_pad_nd_27
# Graph fragment:
#   %constant_pad_nd_27 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_563, [1, 2, 1, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_74 = async_compile.triton('triton_poi_fused_constant_pad_nd_74', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_74', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_74(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 554880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4080) % 17
    x1 = (xindex // 240) % 17
    x0 = xindex % 240
    x3 = (xindex // 69360)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-14160) + x0 + (960*x1) + (13440*x2) + (188160*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ya/cyacnndsjg5hwi7m76vfytabxcwflanyfodftvj7vwheindamqbr.py
# Topologically Sorted Source Nodes: [x_362], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_362 => constant_pad_nd_28
# Graph fragment:
#   %constant_pad_nd_28 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_568, [2, 3, 2, 3], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_75 = async_compile.triton('triton_poi_fused_constant_pad_nd_75', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_75', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_75(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 693120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4560) % 19
    x1 = (xindex // 240) % 19
    x0 = xindex % 240
    x3 = (xindex // 86640)
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-28320) + x0 + (960*x1) + (13440*x2) + (188160*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qs/cqsluywkt7melbpbkqpzm2m2h6346vukgvjv3yhyweane2pmnzsh.py
# Topologically Sorted Source Nodes: [x_363], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_363 => constant_pad_nd_29
# Graph fragment:
#   %constant_pad_nd_29 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%getitem_573, [3, 4, 3, 4], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_76 = async_compile.triton('triton_poi_fused_constant_pad_nd_76', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_76', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_76(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 846720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5040) % 21
    x1 = (xindex // 240) % 21
    x0 = xindex % 240
    x3 = (xindex // 105840)
    x6 = xindex
    tmp0 = (-3) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-42480) + x0 + (960*x1) + (13440*x2) + (188160*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/f4/cf4bbsscnsuk4ab4gofombmhqhxx2aayppo6zzlq5db7e6fsdtww.py
# Topologically Sorted Source Nodes: [x_364, x_365, x_366, x_se_112], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_364 => cat_75
#   x_365 => add_234, mul_426, mul_427, sub_104
#   x_366 => mul_428, sigmoid_113
#   x_se_112 => mean_29
# Graph fragment:
#   %cat_75 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_275, %convolution_276, %convolution_277, %convolution_278], 1), kwargs = {})
#   %sub_104 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_75, %unsqueeze_833), kwargs = {})
#   %mul_426 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_104, %unsqueeze_835), kwargs = {})
#   %mul_427 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_426, %unsqueeze_837), kwargs = {})
#   %add_234 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_427, %unsqueeze_839), kwargs = {})
#   %sigmoid_113 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_234,), kwargs = {})
#   %mul_428 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_234, %sigmoid_113), kwargs = {})
#   %mean_29 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_428, [2, 3], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_77 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_77', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_77', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_77(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 960
    r2 = rindex
    x1 = (xindex // 960)
    x3 = xindex
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 240, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((240*r2) + (11760*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 480, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((240*r2) + (11760*x1) + ((-240) + x0)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1, 1], 720, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + ((240*r2) + (11760*x1) + ((-480) + x0)), rmask & tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1, 1], 960, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + ((240*r2) + (11760*x1) + ((-720) + x0)), rmask & tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 0.001
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tl.full([1, 1], 1, tl.int32)
    tmp30 = tmp29 / tmp28
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tl.sigmoid(tmp37)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.where(rmask & xmask, tmp40, 0)
    tmp43 = tl.sum(tmp42, 1)[:, None]
    tmp44 = 49.0
    tmp45 = tmp43 / tmp44
    tl.store(out_ptr0 + (r2 + (49*x3)), tmp33, rmask & xmask)
    tl.store(out_ptr2 + (x3), tmp45, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/l2/cl2zk7qisvxgs4rx6tlbib3732pi7hzzaccsxi67txm7a6oxaygw.py
# Topologically Sorted Source Nodes: [x_365, x_366, x_se_112, x_se_113, x_se_114, x_se_115, sigmoid_28, x_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_28 => sigmoid_115
#   x_365 => add_234, mul_427
#   x_366 => mul_428, sigmoid_113
#   x_367 => mul_430
#   x_se_112 => mean_29
#   x_se_113 => convolution_279
#   x_se_114 => mul_429, sigmoid_114
#   x_se_115 => convolution_280
# Graph fragment:
#   %mul_427 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_426, %unsqueeze_837), kwargs = {})
#   %add_234 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_427, %unsqueeze_839), kwargs = {})
#   %sigmoid_113 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_234,), kwargs = {})
#   %mul_428 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_234, %sigmoid_113), kwargs = {})
#   %mean_29 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_428, [2, 3], True), kwargs = {})
#   %convolution_279 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_29, %arg337_1, %arg338_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_114 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_279,), kwargs = {})
#   %mul_429 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_279, %sigmoid_114), kwargs = {})
#   %convolution_280 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_429, %arg339_1, %arg340_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_115 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_280,), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_428, %sigmoid_115), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_78 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_78', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_78', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_78(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 960
    y1 = (yindex // 960)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp6 * tmp10
    tl.store(out_ptr0 + (y0 + (960*x2) + (47040*y1)), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4i/c4ircvpwsgdcwcpunzpuomv3fcdgnqkphwaqo6dpafoeqqovnlsr.py
# Topologically Sorted Source Nodes: [x_369], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_369 => add_236, mul_432, mul_433, sub_105
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_281, %unsqueeze_841), kwargs = {})
#   %mul_432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_432, %unsqueeze_845), kwargs = {})
#   %add_236 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_433, %unsqueeze_847), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_79 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_79', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_79', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_79(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 103488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 264
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


# kernel path: /tmp/torchinductor_sahanp/fn/cfn4uixbxpyme3hwcp25o5eqrzvgnbf7zny63aivl2wpakagdh7j.py
# Topologically Sorted Source Nodes: [x_371, x_372], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_371 => add_238, mul_435, mul_436, sub_106
#   x_372 => mul_437, sigmoid_116
# Graph fragment:
#   %sub_106 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_282, %unsqueeze_849), kwargs = {})
#   %mul_435 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_106, %unsqueeze_851), kwargs = {})
#   %mul_436 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_435, %unsqueeze_853), kwargs = {})
#   %add_238 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_436, %unsqueeze_855), kwargs = {})
#   %sigmoid_116 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_238,), kwargs = {})
#   %mul_437 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_238, %sigmoid_116), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_80 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_80', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_80', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_80(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 1584
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 49
    y3 = (yindex // 49)
    tmp0 = tl.load(in_out_ptr0 + (x1 + (1584*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (y2 + (49*x1) + (77616*y3)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n2/cn2z6lghkpkszq3pcdmnznsfkxicxnkwzvbncbd2me76uibjqxoz.py
# Topologically Sorted Source Nodes: [conv2d_283], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_283 => convolution_283
# Graph fragment:
#   %convolution_283 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_578, %arg351_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396), kwargs = {})
triton_poi_fused_convolution_81 = async_compile.triton('triton_poi_fused_convolution_81', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_81', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_81(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 396
    y1 = (yindex // 396)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ky/ckyrdjo7ajl6docs2honlmbcn2fk5gk3lr3lrejolh5vcaymhxgr.py
# Topologically Sorted Source Nodes: [conv2d_284], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_284 => convolution_284
# Graph fragment:
#   %convolution_284 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_583, %arg352_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396), kwargs = {})
triton_poi_fused_convolution_82 = async_compile.triton('triton_poi_fused_convolution_82', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_82', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_82(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 396
    y1 = (yindex // 396)
    tmp0 = tl.load(in_ptr0 + (19404 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oc/coc5rubfqaxekvtd4fpdn3zkgcrtngv3tuzqnyhbqlvd4mtiiv6v.py
# Topologically Sorted Source Nodes: [conv2d_285], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_285 => convolution_285
# Graph fragment:
#   %convolution_285 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_588, %arg353_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396), kwargs = {})
triton_poi_fused_convolution_83 = async_compile.triton('triton_poi_fused_convolution_83', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_83', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_83(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 396
    y1 = (yindex // 396)
    tmp0 = tl.load(in_ptr0 + (38808 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gm/cgmuethkmycpzrfpw44zlmygwzsht2qxaetyhm23zrdfnojgxptx.py
# Topologically Sorted Source Nodes: [conv2d_286], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_286 => convolution_286
# Graph fragment:
#   %convolution_286 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_593, %arg354_1, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396), kwargs = {})
triton_poi_fused_convolution_84 = async_compile.triton('triton_poi_fused_convolution_84', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_84', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_84(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 396
    y1 = (yindex // 396)
    tmp0 = tl.load(in_ptr0 + (58212 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6d/c6dncc6zhd6eojaxhvuczcvued6kevfjngeoygogjuqhfourw7y7.py
# Topologically Sorted Source Nodes: [x_373, x_374, x_375, x_se_116], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_373 => cat_76
#   x_374 => add_240, mul_439, mul_440, sub_107
#   x_375 => mul_441, sigmoid_117
#   x_se_116 => mean_30
# Graph fragment:
#   %cat_76 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_283, %convolution_284, %convolution_285, %convolution_286], 1), kwargs = {})
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_76, %unsqueeze_857), kwargs = {})
#   %mul_439 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %unsqueeze_859), kwargs = {})
#   %mul_440 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_439, %unsqueeze_861), kwargs = {})
#   %add_240 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_440, %unsqueeze_863), kwargs = {})
#   %sigmoid_117 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_240,), kwargs = {})
#   %mul_441 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_240, %sigmoid_117), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_441, [2, 3], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_85 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_85', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_85', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_85(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12672
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 1584
    r2 = rindex
    x1 = (xindex // 1584)
    x3 = xindex
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 396, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((396*r2) + (19404*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 792, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((396*r2) + (19404*x1) + ((-396) + x0)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1, 1], 1188, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + ((396*r2) + (19404*x1) + ((-792) + x0)), rmask & tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1, 1], 1584, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + ((396*r2) + (19404*x1) + ((-1188) + x0)), rmask & tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 0.001
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tl.full([1, 1], 1, tl.int32)
    tmp30 = tmp29 / tmp28
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tl.sigmoid(tmp37)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.where(rmask & xmask, tmp40, 0)
    tmp43 = tl.sum(tmp42, 1)[:, None]
    tmp44 = 49.0
    tmp45 = tmp43 / tmp44
    tl.store(out_ptr0 + (r2 + (49*x0) + (77632*x1)), tmp33, rmask & xmask)
    tl.store(out_ptr2 + (x3), tmp45, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rp/crpx3ifwzmarjsdnckldsub3eudabmqnsrh7hdqyb5ficy5wiwxz.py
# Topologically Sorted Source Nodes: [x_374, x_375, x_se_116, x_se_117, x_se_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_374 => add_240, mul_440
#   x_375 => mul_441, sigmoid_117
#   x_se_116 => mean_30
#   x_se_117 => convolution_287
#   x_se_118 => mul_442, sigmoid_118
# Graph fragment:
#   %mul_440 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_439, %unsqueeze_861), kwargs = {})
#   %add_240 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_440, %unsqueeze_863), kwargs = {})
#   %sigmoid_117 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_240,), kwargs = {})
#   %mul_441 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_240, %sigmoid_117), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_441, [2, 3], True), kwargs = {})
#   %convolution_287 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_30, %arg359_1, %arg360_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_118 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_287,), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_287, %sigmoid_118), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_86 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_86', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_86', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_86(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 132
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/62/c62qcz4aqbk4riv3deijr7anupweoxddlyac2futehl6qa7cpcyn.py
# Topologically Sorted Source Nodes: [x_374, x_375, x_se_116, x_se_117, x_se_118, x_se_119, sigmoid_29, x_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_29 => sigmoid_119
#   x_374 => add_240, mul_440
#   x_375 => mul_441, sigmoid_117
#   x_376 => mul_443
#   x_se_116 => mean_30
#   x_se_117 => convolution_287
#   x_se_118 => mul_442, sigmoid_118
#   x_se_119 => convolution_288
# Graph fragment:
#   %mul_440 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_439, %unsqueeze_861), kwargs = {})
#   %add_240 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_440, %unsqueeze_863), kwargs = {})
#   %sigmoid_117 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_240,), kwargs = {})
#   %mul_441 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_240, %sigmoid_117), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_441, [2, 3], True), kwargs = {})
#   %convolution_287 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_30, %arg359_1, %arg360_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_118 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_287,), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_287, %sigmoid_118), kwargs = {})
#   %convolution_288 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_442, %arg361_1, %arg362_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_119 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_288,), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_441, %sigmoid_119), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_87 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_87', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_87', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_87(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 620928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 77616)
    x3 = xindex % 77616
    x1 = (xindex // 49) % 1584
    x4 = (xindex // 49)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (77632*x2)), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp6 * tmp10
    tl.store(out_ptr0 + (x5), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lt/cltsykeyofv64pai6pf3kv3l7gsqua6s3yiqdhhwufksapwmhbz3.py
# Topologically Sorted Source Nodes: [conv2d_289], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_289 => convolution_289
# Graph fragment:
#   %convolution_289 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_594, %arg363_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_88 = async_compile.triton('triton_poi_fused_convolution_88', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_88', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_88(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6336
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 792
    y1 = (yindex // 792)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (792*x2) + (38808*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rr/crrvn2qyuhwjglfrj4n7wkatzn2w3ctlroohhc46s7wk7uv3vcli.py
# Topologically Sorted Source Nodes: [conv2d_290], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_290 => convolution_290
# Graph fragment:
#   %convolution_290 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_595, %arg364_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_89 = async_compile.triton('triton_poi_fused_convolution_89', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_89', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_89(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6336
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 792
    y1 = (yindex // 792)
    tmp0 = tl.load(in_ptr0 + (38808 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (792*x2) + (38808*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wu/cwucpsi736t2lfhrdrde5nqknluyx7jgxgo7uslo4lopcneifhth.py
# Topologically Sorted Source Nodes: [x_377, x_378, x_379], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_377 => cat_77
#   x_378 => add_242, mul_445, mul_446, sub_108
#   x_379 => add_243
# Graph fragment:
#   %cat_77 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_289, %convolution_290], 1), kwargs = {})
#   %sub_108 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_77, %unsqueeze_865), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_108, %unsqueeze_867), kwargs = {})
#   %mul_446 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_445, %unsqueeze_869), kwargs = {})
#   %add_242 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_446, %unsqueeze_871), kwargs = {})
#   %add_243 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_242, %add_236), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_90 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_90', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_90', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_90(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 103488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 264
    x1 = (xindex // 264)
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 132, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((132*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 264, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((132*x1) + ((-132) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp27 = tmp25 + tmp26
    tl.store(in_out_ptr0 + (x2), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ae/caeqh6t2asw6ldip7mnm7jpmvzegv6vg2vynwkpjtd4nkpreez6u.py
# Topologically Sorted Source Nodes: [x_401, x_402, x_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   x_401 => add_259, mul_474, mul_475, sub_115
#   x_402 => relu_13
#   x_403 => mean_33
# Graph fragment:
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_309, %unsqueeze_921), kwargs = {})
#   %mul_474 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %unsqueeze_923), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_474, %unsqueeze_925), kwargs = {})
#   %add_259 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_475, %unsqueeze_927), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_259,), kwargs = {})
#   %mean_33 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_13, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_91 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_mean_relu_91', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_91', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_mean_relu_91(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1536
    x1 = (xindex // 1536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*r2) + (75264*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 49.0
    tmp23 = tmp21 / tmp22
    tl.store(out_ptr1 + (x3), tmp23, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (32, ), (1, ))
    assert_size_stride(arg9_1, (32, ), (1, ))
    assert_size_stride(arg10_1, (32, ), (1, ))
    assert_size_stride(arg11_1, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg12_1, (32, ), (1, ))
    assert_size_stride(arg13_1, (32, ), (1, ))
    assert_size_stride(arg14_1, (32, ), (1, ))
    assert_size_stride(arg15_1, (32, ), (1, ))
    assert_size_stride(arg16_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg17_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg18_1, (192, ), (1, ))
    assert_size_stride(arg19_1, (192, ), (1, ))
    assert_size_stride(arg20_1, (192, ), (1, ))
    assert_size_stride(arg21_1, (192, ), (1, ))
    assert_size_stride(arg22_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg23_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg24_1, (64, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg25_1, (192, ), (1, ))
    assert_size_stride(arg26_1, (192, ), (1, ))
    assert_size_stride(arg27_1, (192, ), (1, ))
    assert_size_stride(arg28_1, (192, ), (1, ))
    assert_size_stride(arg29_1, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg30_1, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg31_1, (40, ), (1, ))
    assert_size_stride(arg32_1, (40, ), (1, ))
    assert_size_stride(arg33_1, (40, ), (1, ))
    assert_size_stride(arg34_1, (40, ), (1, ))
    assert_size_stride(arg35_1, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg36_1, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg37_1, (120, ), (1, ))
    assert_size_stride(arg38_1, (120, ), (1, ))
    assert_size_stride(arg39_1, (120, ), (1, ))
    assert_size_stride(arg40_1, (120, ), (1, ))
    assert_size_stride(arg41_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg42_1, (120, ), (1, ))
    assert_size_stride(arg43_1, (120, ), (1, ))
    assert_size_stride(arg44_1, (120, ), (1, ))
    assert_size_stride(arg45_1, (120, ), (1, ))
    assert_size_stride(arg46_1, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(arg47_1, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(arg48_1, (40, ), (1, ))
    assert_size_stride(arg49_1, (40, ), (1, ))
    assert_size_stride(arg50_1, (40, ), (1, ))
    assert_size_stride(arg51_1, (40, ), (1, ))
    assert_size_stride(arg52_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg53_1, (240, ), (1, ))
    assert_size_stride(arg54_1, (240, ), (1, ))
    assert_size_stride(arg55_1, (240, ), (1, ))
    assert_size_stride(arg56_1, (240, ), (1, ))
    assert_size_stride(arg57_1, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg58_1, (60, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg59_1, (60, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg60_1, (60, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg61_1, (240, ), (1, ))
    assert_size_stride(arg62_1, (240, ), (1, ))
    assert_size_stride(arg63_1, (240, ), (1, ))
    assert_size_stride(arg64_1, (240, ), (1, ))
    assert_size_stride(arg65_1, (20, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg66_1, (20, ), (1, ))
    assert_size_stride(arg67_1, (240, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg68_1, (240, ), (1, ))
    assert_size_stride(arg69_1, (56, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg70_1, (56, ), (1, ))
    assert_size_stride(arg71_1, (56, ), (1, ))
    assert_size_stride(arg72_1, (56, ), (1, ))
    assert_size_stride(arg73_1, (56, ), (1, ))
    assert_size_stride(arg74_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg75_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg76_1, (336, ), (1, ))
    assert_size_stride(arg77_1, (336, ), (1, ))
    assert_size_stride(arg78_1, (336, ), (1, ))
    assert_size_stride(arg79_1, (336, ), (1, ))
    assert_size_stride(arg80_1, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg81_1, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg82_1, (336, ), (1, ))
    assert_size_stride(arg83_1, (336, ), (1, ))
    assert_size_stride(arg84_1, (336, ), (1, ))
    assert_size_stride(arg85_1, (336, ), (1, ))
    assert_size_stride(arg86_1, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg87_1, (28, ), (1, ))
    assert_size_stride(arg88_1, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg89_1, (336, ), (1, ))
    assert_size_stride(arg90_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg91_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg92_1, (56, ), (1, ))
    assert_size_stride(arg93_1, (56, ), (1, ))
    assert_size_stride(arg94_1, (56, ), (1, ))
    assert_size_stride(arg95_1, (56, ), (1, ))
    assert_size_stride(arg96_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg97_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg98_1, (336, ), (1, ))
    assert_size_stride(arg99_1, (336, ), (1, ))
    assert_size_stride(arg100_1, (336, ), (1, ))
    assert_size_stride(arg101_1, (336, ), (1, ))
    assert_size_stride(arg102_1, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg103_1, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg104_1, (336, ), (1, ))
    assert_size_stride(arg105_1, (336, ), (1, ))
    assert_size_stride(arg106_1, (336, ), (1, ))
    assert_size_stride(arg107_1, (336, ), (1, ))
    assert_size_stride(arg108_1, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg109_1, (28, ), (1, ))
    assert_size_stride(arg110_1, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg111_1, (336, ), (1, ))
    assert_size_stride(arg112_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg113_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg114_1, (56, ), (1, ))
    assert_size_stride(arg115_1, (56, ), (1, ))
    assert_size_stride(arg116_1, (56, ), (1, ))
    assert_size_stride(arg117_1, (56, ), (1, ))
    assert_size_stride(arg118_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg119_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg120_1, (336, ), (1, ))
    assert_size_stride(arg121_1, (336, ), (1, ))
    assert_size_stride(arg122_1, (336, ), (1, ))
    assert_size_stride(arg123_1, (336, ), (1, ))
    assert_size_stride(arg124_1, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg125_1, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg126_1, (336, ), (1, ))
    assert_size_stride(arg127_1, (336, ), (1, ))
    assert_size_stride(arg128_1, (336, ), (1, ))
    assert_size_stride(arg129_1, (336, ), (1, ))
    assert_size_stride(arg130_1, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg131_1, (28, ), (1, ))
    assert_size_stride(arg132_1, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg133_1, (336, ), (1, ))
    assert_size_stride(arg134_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg135_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg136_1, (56, ), (1, ))
    assert_size_stride(arg137_1, (56, ), (1, ))
    assert_size_stride(arg138_1, (56, ), (1, ))
    assert_size_stride(arg139_1, (56, ), (1, ))
    assert_size_stride(arg140_1, (336, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg141_1, (336, ), (1, ))
    assert_size_stride(arg142_1, (336, ), (1, ))
    assert_size_stride(arg143_1, (336, ), (1, ))
    assert_size_stride(arg144_1, (336, ), (1, ))
    assert_size_stride(arg145_1, (112, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg146_1, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg147_1, (112, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg148_1, (336, ), (1, ))
    assert_size_stride(arg149_1, (336, ), (1, ))
    assert_size_stride(arg150_1, (336, ), (1, ))
    assert_size_stride(arg151_1, (336, ), (1, ))
    assert_size_stride(arg152_1, (14, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg153_1, (14, ), (1, ))
    assert_size_stride(arg154_1, (336, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(arg155_1, (336, ), (1, ))
    assert_size_stride(arg156_1, (104, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg157_1, (104, ), (1, ))
    assert_size_stride(arg158_1, (104, ), (1, ))
    assert_size_stride(arg159_1, (104, ), (1, ))
    assert_size_stride(arg160_1, (104, ), (1, ))
    assert_size_stride(arg161_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg162_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg163_1, (624, ), (1, ))
    assert_size_stride(arg164_1, (624, ), (1, ))
    assert_size_stride(arg165_1, (624, ), (1, ))
    assert_size_stride(arg166_1, (624, ), (1, ))
    assert_size_stride(arg167_1, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg168_1, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg169_1, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg170_1, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg171_1, (624, ), (1, ))
    assert_size_stride(arg172_1, (624, ), (1, ))
    assert_size_stride(arg173_1, (624, ), (1, ))
    assert_size_stride(arg174_1, (624, ), (1, ))
    assert_size_stride(arg175_1, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg176_1, (26, ), (1, ))
    assert_size_stride(arg177_1, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(arg178_1, (624, ), (1, ))
    assert_size_stride(arg179_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg180_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg181_1, (104, ), (1, ))
    assert_size_stride(arg182_1, (104, ), (1, ))
    assert_size_stride(arg183_1, (104, ), (1, ))
    assert_size_stride(arg184_1, (104, ), (1, ))
    assert_size_stride(arg185_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg186_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg187_1, (624, ), (1, ))
    assert_size_stride(arg188_1, (624, ), (1, ))
    assert_size_stride(arg189_1, (624, ), (1, ))
    assert_size_stride(arg190_1, (624, ), (1, ))
    assert_size_stride(arg191_1, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg192_1, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg193_1, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg194_1, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg195_1, (624, ), (1, ))
    assert_size_stride(arg196_1, (624, ), (1, ))
    assert_size_stride(arg197_1, (624, ), (1, ))
    assert_size_stride(arg198_1, (624, ), (1, ))
    assert_size_stride(arg199_1, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg200_1, (26, ), (1, ))
    assert_size_stride(arg201_1, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(arg202_1, (624, ), (1, ))
    assert_size_stride(arg203_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg204_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg205_1, (104, ), (1, ))
    assert_size_stride(arg206_1, (104, ), (1, ))
    assert_size_stride(arg207_1, (104, ), (1, ))
    assert_size_stride(arg208_1, (104, ), (1, ))
    assert_size_stride(arg209_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg210_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg211_1, (624, ), (1, ))
    assert_size_stride(arg212_1, (624, ), (1, ))
    assert_size_stride(arg213_1, (624, ), (1, ))
    assert_size_stride(arg214_1, (624, ), (1, ))
    assert_size_stride(arg215_1, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg216_1, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg217_1, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg218_1, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg219_1, (624, ), (1, ))
    assert_size_stride(arg220_1, (624, ), (1, ))
    assert_size_stride(arg221_1, (624, ), (1, ))
    assert_size_stride(arg222_1, (624, ), (1, ))
    assert_size_stride(arg223_1, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg224_1, (26, ), (1, ))
    assert_size_stride(arg225_1, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(arg226_1, (624, ), (1, ))
    assert_size_stride(arg227_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg228_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg229_1, (104, ), (1, ))
    assert_size_stride(arg230_1, (104, ), (1, ))
    assert_size_stride(arg231_1, (104, ), (1, ))
    assert_size_stride(arg232_1, (104, ), (1, ))
    assert_size_stride(arg233_1, (624, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(arg234_1, (624, ), (1, ))
    assert_size_stride(arg235_1, (624, ), (1, ))
    assert_size_stride(arg236_1, (624, ), (1, ))
    assert_size_stride(arg237_1, (624, ), (1, ))
    assert_size_stride(arg238_1, (624, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg239_1, (624, ), (1, ))
    assert_size_stride(arg240_1, (624, ), (1, ))
    assert_size_stride(arg241_1, (624, ), (1, ))
    assert_size_stride(arg242_1, (624, ), (1, ))
    assert_size_stride(arg243_1, (52, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg244_1, (52, ), (1, ))
    assert_size_stride(arg245_1, (624, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg246_1, (624, ), (1, ))
    assert_size_stride(arg247_1, (160, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg248_1, (160, ), (1, ))
    assert_size_stride(arg249_1, (160, ), (1, ))
    assert_size_stride(arg250_1, (160, ), (1, ))
    assert_size_stride(arg251_1, (160, ), (1, ))
    assert_size_stride(arg252_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg253_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg254_1, (480, ), (1, ))
    assert_size_stride(arg255_1, (480, ), (1, ))
    assert_size_stride(arg256_1, (480, ), (1, ))
    assert_size_stride(arg257_1, (480, ), (1, ))
    assert_size_stride(arg258_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg259_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg260_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg261_1, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg262_1, (480, ), (1, ))
    assert_size_stride(arg263_1, (480, ), (1, ))
    assert_size_stride(arg264_1, (480, ), (1, ))
    assert_size_stride(arg265_1, (480, ), (1, ))
    assert_size_stride(arg266_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg267_1, (80, ), (1, ))
    assert_size_stride(arg268_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg269_1, (480, ), (1, ))
    assert_size_stride(arg270_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg271_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg272_1, (160, ), (1, ))
    assert_size_stride(arg273_1, (160, ), (1, ))
    assert_size_stride(arg274_1, (160, ), (1, ))
    assert_size_stride(arg275_1, (160, ), (1, ))
    assert_size_stride(arg276_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg277_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg278_1, (480, ), (1, ))
    assert_size_stride(arg279_1, (480, ), (1, ))
    assert_size_stride(arg280_1, (480, ), (1, ))
    assert_size_stride(arg281_1, (480, ), (1, ))
    assert_size_stride(arg282_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg283_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg284_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg285_1, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg286_1, (480, ), (1, ))
    assert_size_stride(arg287_1, (480, ), (1, ))
    assert_size_stride(arg288_1, (480, ), (1, ))
    assert_size_stride(arg289_1, (480, ), (1, ))
    assert_size_stride(arg290_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg291_1, (80, ), (1, ))
    assert_size_stride(arg292_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg293_1, (480, ), (1, ))
    assert_size_stride(arg294_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg295_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg296_1, (160, ), (1, ))
    assert_size_stride(arg297_1, (160, ), (1, ))
    assert_size_stride(arg298_1, (160, ), (1, ))
    assert_size_stride(arg299_1, (160, ), (1, ))
    assert_size_stride(arg300_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg301_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg302_1, (480, ), (1, ))
    assert_size_stride(arg303_1, (480, ), (1, ))
    assert_size_stride(arg304_1, (480, ), (1, ))
    assert_size_stride(arg305_1, (480, ), (1, ))
    assert_size_stride(arg306_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg307_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg308_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg309_1, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg310_1, (480, ), (1, ))
    assert_size_stride(arg311_1, (480, ), (1, ))
    assert_size_stride(arg312_1, (480, ), (1, ))
    assert_size_stride(arg313_1, (480, ), (1, ))
    assert_size_stride(arg314_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg315_1, (80, ), (1, ))
    assert_size_stride(arg316_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg317_1, (480, ), (1, ))
    assert_size_stride(arg318_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg319_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg320_1, (160, ), (1, ))
    assert_size_stride(arg321_1, (160, ), (1, ))
    assert_size_stride(arg322_1, (160, ), (1, ))
    assert_size_stride(arg323_1, (160, ), (1, ))
    assert_size_stride(arg324_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg325_1, (960, ), (1, ))
    assert_size_stride(arg326_1, (960, ), (1, ))
    assert_size_stride(arg327_1, (960, ), (1, ))
    assert_size_stride(arg328_1, (960, ), (1, ))
    assert_size_stride(arg329_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg330_1, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg331_1, (240, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg332_1, (240, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg333_1, (960, ), (1, ))
    assert_size_stride(arg334_1, (960, ), (1, ))
    assert_size_stride(arg335_1, (960, ), (1, ))
    assert_size_stride(arg336_1, (960, ), (1, ))
    assert_size_stride(arg337_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg338_1, (80, ), (1, ))
    assert_size_stride(arg339_1, (960, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg340_1, (960, ), (1, ))
    assert_size_stride(arg341_1, (264, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg342_1, (264, ), (1, ))
    assert_size_stride(arg343_1, (264, ), (1, ))
    assert_size_stride(arg344_1, (264, ), (1, ))
    assert_size_stride(arg345_1, (264, ), (1, ))
    assert_size_stride(arg346_1, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(arg347_1, (1584, ), (1, ))
    assert_size_stride(arg348_1, (1584, ), (1, ))
    assert_size_stride(arg349_1, (1584, ), (1, ))
    assert_size_stride(arg350_1, (1584, ), (1, ))
    assert_size_stride(arg351_1, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg352_1, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg353_1, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg354_1, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg355_1, (1584, ), (1, ))
    assert_size_stride(arg356_1, (1584, ), (1, ))
    assert_size_stride(arg357_1, (1584, ), (1, ))
    assert_size_stride(arg358_1, (1584, ), (1, ))
    assert_size_stride(arg359_1, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(arg360_1, (132, ), (1, ))
    assert_size_stride(arg361_1, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(arg362_1, (1584, ), (1, ))
    assert_size_stride(arg363_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg364_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg365_1, (264, ), (1, ))
    assert_size_stride(arg366_1, (264, ), (1, ))
    assert_size_stride(arg367_1, (264, ), (1, ))
    assert_size_stride(arg368_1, (264, ), (1, ))
    assert_size_stride(arg369_1, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(arg370_1, (1584, ), (1, ))
    assert_size_stride(arg371_1, (1584, ), (1, ))
    assert_size_stride(arg372_1, (1584, ), (1, ))
    assert_size_stride(arg373_1, (1584, ), (1, ))
    assert_size_stride(arg374_1, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg375_1, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg376_1, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg377_1, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg378_1, (1584, ), (1, ))
    assert_size_stride(arg379_1, (1584, ), (1, ))
    assert_size_stride(arg380_1, (1584, ), (1, ))
    assert_size_stride(arg381_1, (1584, ), (1, ))
    assert_size_stride(arg382_1, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(arg383_1, (132, ), (1, ))
    assert_size_stride(arg384_1, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(arg385_1, (1584, ), (1, ))
    assert_size_stride(arg386_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg387_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg388_1, (264, ), (1, ))
    assert_size_stride(arg389_1, (264, ), (1, ))
    assert_size_stride(arg390_1, (264, ), (1, ))
    assert_size_stride(arg391_1, (264, ), (1, ))
    assert_size_stride(arg392_1, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(arg393_1, (1584, ), (1, ))
    assert_size_stride(arg394_1, (1584, ), (1, ))
    assert_size_stride(arg395_1, (1584, ), (1, ))
    assert_size_stride(arg396_1, (1584, ), (1, ))
    assert_size_stride(arg397_1, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg398_1, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg399_1, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg400_1, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg401_1, (1584, ), (1, ))
    assert_size_stride(arg402_1, (1584, ), (1, ))
    assert_size_stride(arg403_1, (1584, ), (1, ))
    assert_size_stride(arg404_1, (1584, ), (1, ))
    assert_size_stride(arg405_1, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(arg406_1, (132, ), (1, ))
    assert_size_stride(arg407_1, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(arg408_1, (1584, ), (1, ))
    assert_size_stride(arg409_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg410_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg411_1, (264, ), (1, ))
    assert_size_stride(arg412_1, (264, ), (1, ))
    assert_size_stride(arg413_1, (264, ), (1, ))
    assert_size_stride(arg414_1, (264, ), (1, ))
    assert_size_stride(arg415_1, (1536, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(arg416_1, (1536, ), (1, ))
    assert_size_stride(arg417_1, (1536, ), (1, ))
    assert_size_stride(arg418_1, (1536, ), (1, ))
    assert_size_stride(arg419_1, (1536, ), (1, ))
    assert_size_stride(arg420_1, (1000, 1536), (1536, 1))
    assert_size_stride(arg421_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 225, 225), (151875, 1, 675, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_203], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0.run(arg1_1, buf0, 24, 50625, grid=grid(24, 50625), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_203, x_204], Original ATen: [aten.constant_pad_nd, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_203, x_204], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        # Topologically Sorted Source Nodes: [x_207], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf4, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del arg6_1
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_208, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf5, arg7_1, arg8_1, arg9_1, arg10_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [x_208, x_209, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del arg11_1
        buf7 = reinterpret_tensor(buf5, (8, 32, 112, 112), (401408, 12544, 112, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_211, x_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_3.run(buf6, arg12_1, arg13_1, arg14_1, arg15_1, buf3, buf7, 100352, 32, grid=grid(100352, 32), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del buf3
        del buf6
        buf8 = empty_strided_cuda((8, 16, 112, 112), (200704, 1, 1792, 16), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_158], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_4.run(buf7, buf8, 128, 12544, grid=grid(128, 12544), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_158], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 96, 112, 112), (1204224, 1, 10752, 96))
        del arg16_1
        buf10 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [conv2d_159], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(buf7, buf10, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del buf7
        # Topologically Sorted Source Nodes: [conv2d_159], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg17_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 96, 112, 112), (1204224, 1, 10752, 96))
        del arg17_1
        del buf10
        buf12 = empty_strided_cuda((8, 192, 112, 112), (2408448, 12544, 112, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_213, x_214, x_215], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6.run(buf9, buf11, arg18_1, arg19_1, arg20_1, arg21_1, buf12, 19267584, grid=grid(19267584), stream=stream0)
        del arg18_1
        del arg19_1
        del arg20_1
        del arg21_1
        del buf11
        del buf9
        buf13 = empty_strided_cuda((8, 64, 113, 113), (817216, 1, 7232, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_216], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_7.run(buf12, buf13, 512, 12769, grid=grid(512, 12769), stream=stream0)
        # Topologically Sorted Source Nodes: [x_216, conv2d_160], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg22_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf14, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg22_1
        del buf13
        buf15 = empty_strided_cuda((8, 64, 115, 115), (846400, 1, 7360, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_217], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_8.run(buf12, buf15, 512, 13225, grid=grid(512, 13225), stream=stream0)
        # Topologically Sorted Source Nodes: [x_217, conv2d_161], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg23_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf16, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg23_1
        del buf15
        buf17 = empty_strided_cuda((8, 64, 117, 117), (876096, 1, 7488, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_218], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_9.run(buf12, buf17, 512, 13689, grid=grid(512, 13689), stream=stream0)
        del buf12
        # Topologically Sorted Source Nodes: [x_218, conv2d_162], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg24_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf18, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg24_1
        del buf17
        buf20 = empty_strided_cuda((8, 192, 56, 56), (602112, 3136, 56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_219, x_220, x_221], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10.run(buf14, buf16, buf18, arg25_1, arg26_1, arg27_1, arg28_1, buf20, 4816896, grid=grid(4816896), stream=stream0)
        del arg25_1
        del arg26_1
        del arg27_1
        del arg28_1
        del buf14
        del buf16
        del buf18
        buf21 = empty_strided_cuda((8, 96, 56, 56), (301056, 1, 5376, 96), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_163], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(buf20, buf21, 768, 3136, grid=grid(768, 3136), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_163], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg29_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 20, 56, 56), (62720, 1, 1120, 20))
        del arg29_1
        buf23 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [conv2d_164], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf20, buf23, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del buf20
        # Topologically Sorted Source Nodes: [conv2d_164], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 20, 56, 56), (62720, 1, 1120, 20))
        del arg30_1
        del buf23
        buf25 = empty_strided_cuda((8, 40, 56, 56), (125440, 3136, 56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_222, x_223], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_13.run(buf22, buf24, arg31_1, arg32_1, arg33_1, arg34_1, buf25, 1003520, grid=grid(1003520), stream=stream0)
        del arg31_1
        del arg32_1
        del arg33_1
        del arg34_1
        del buf22
        buf26 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [conv2d_165], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf25, buf26, 160, 3136, grid=grid(160, 3136), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_165], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, arg35_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 60, 56, 56), (188160, 1, 3360, 60))
        del arg35_1
        buf28 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [conv2d_166], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(buf25, buf28, 160, 3136, grid=grid(160, 3136), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_166], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 60, 56, 56), (188160, 1, 3360, 60))
        del arg36_1
        del buf28
        buf30 = empty_strided_cuda((8, 120, 56, 56), (376320, 1, 6720, 120), torch.float32)
        # Topologically Sorted Source Nodes: [x_224, x_225, x_226], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf27, buf29, arg37_1, arg38_1, arg39_1, arg40_1, buf30, 3010560, grid=grid(3010560), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        # Topologically Sorted Source Nodes: [x_224, x_225, x_226, x_227], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg41_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf31, (8, 120, 56, 56), (376320, 1, 6720, 120))
        del arg41_1
        del buf30
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_228, x_229], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf32, arg42_1, arg43_1, arg44_1, arg45_1, 3010560, grid=grid(3010560), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        # Topologically Sorted Source Nodes: [conv2d_168], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(reinterpret_tensor(buf32, (8, 60, 56, 56), (376320, 1, 6720, 120), 0), arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 20, 56, 56), (62720, 1, 1120, 20))
        del arg46_1
        # Topologically Sorted Source Nodes: [conv2d_169], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(reinterpret_tensor(buf32, (8, 60, 56, 56), (376320, 1, 6720, 120), 60), arg47_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 20, 56, 56), (62720, 1, 1120, 20))
        del arg47_1
        del buf32
        buf35 = empty_strided_cuda((8, 40, 56, 56), (125440, 1, 2240, 40), torch.float32)
        # Topologically Sorted Source Nodes: [x_230, x_231, x_232], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_18.run(buf33, buf34, arg48_1, arg49_1, arg50_1, arg51_1, buf25, buf35, 320, 3136, grid=grid(320, 3136), stream=stream0)
        del arg48_1
        del arg49_1
        del arg50_1
        del arg51_1
        del buf25
        del buf33
        del buf34
        # Topologically Sorted Source Nodes: [x_230, x_231, x_232, x_233], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg52_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 240, 56, 56), (752640, 1, 13440, 240))
        del arg52_1
        del buf35
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_234], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf37, arg53_1, arg54_1, arg55_1, arg56_1, 6021120, grid=grid(6021120), stream=stream0)
        del arg53_1
        del arg54_1
        del arg55_1
        del arg56_1
        buf38 = empty_strided_cuda((8, 60, 57, 57), (194940, 1, 3420, 60), torch.float32)
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_20.run(buf37, buf38, 1559520, grid=grid(1559520), stream=stream0)
        # Topologically Sorted Source Nodes: [x_236, conv2d_171], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf39 = extern_kernels.convolution(buf38, arg57_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf39, (8, 60, 28, 28), (47040, 1, 1680, 60))
        del arg57_1
        del buf38
        buf40 = empty_strided_cuda((8, 60, 59, 59), (208860, 1, 3540, 60), torch.float32)
        # Topologically Sorted Source Nodes: [x_237], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_21.run(buf37, buf40, 1670880, grid=grid(1670880), stream=stream0)
        # Topologically Sorted Source Nodes: [x_237, conv2d_172], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg58_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf41, (8, 60, 28, 28), (47040, 1, 1680, 60))
        del arg58_1
        del buf40
        buf42 = empty_strided_cuda((8, 60, 61, 61), (223260, 1, 3660, 60), torch.float32)
        # Topologically Sorted Source Nodes: [x_238], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_22.run(buf37, buf42, 1786080, grid=grid(1786080), stream=stream0)
        # Topologically Sorted Source Nodes: [x_238, conv2d_173], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf43 = extern_kernels.convolution(buf42, arg59_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf43, (8, 60, 28, 28), (47040, 1, 1680, 60))
        del arg59_1
        del buf42
        buf44 = empty_strided_cuda((8, 60, 63, 63), (238140, 1, 3780, 60), torch.float32)
        # Topologically Sorted Source Nodes: [x_239], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_23.run(buf37, buf44, 1905120, grid=grid(1905120), stream=stream0)
        del buf37
        # Topologically Sorted Source Nodes: [x_239, conv2d_174], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg60_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf45, (8, 60, 28, 28), (47040, 1, 1680, 60))
        del arg60_1
        del buf44
        buf46 = reinterpret_tensor(buf29, (8, 240, 28, 28), (188160, 784, 28, 1), 0); del buf29  # reuse
        buf48 = empty_strided_cuda((8, 240, 1, 1), (240, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_240, x_241, x_242, x_se_64], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_24.run(buf39, buf41, buf43, buf45, arg61_1, arg62_1, arg63_1, arg64_1, buf46, buf48, 1920, 784, grid=grid(1920), stream=stream0)
        del arg61_1
        del arg62_1
        del buf39
        del buf41
        del buf43
        del buf45
        # Topologically Sorted Source Nodes: [x_241, x_242, x_se_64, x_se_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg65_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg65_1
        del buf48
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_242, x_se_64, x_se_65, x_se_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_25.run(buf50, arg66_1, 160, grid=grid(160), stream=stream0)
        del arg66_1
        # Topologically Sorted Source Nodes: [x_241, x_242, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg67_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg67_1
        del buf50
        buf52 = reinterpret_tensor(buf27, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_242, x_se_64, x_se_65, x_se_66, x_se_67, sigmoid_16, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_26.run(buf46, arg63_1, arg64_1, buf51, arg68_1, buf52, 1920, 784, grid=grid(1920, 784), stream=stream0)
        del arg63_1
        del arg64_1
        del arg68_1
        del buf46
        del buf51
        # Topologically Sorted Source Nodes: [x_241, x_242, x_se_64, x_se_65, x_se_66, x_se_67, sigmoid_16, x_243, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf53 = extern_kernels.convolution(buf52, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (8, 56, 28, 28), (43904, 1, 1568, 56))
        del arg69_1
        del buf52
        buf54 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf54, arg70_1, arg71_1, arg72_1, arg73_1, 351232, grid=grid(351232), stream=stream0)
        del arg70_1
        del arg71_1
        del arg72_1
        del arg73_1
        # Topologically Sorted Source Nodes: [conv2d_178], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(reinterpret_tensor(buf54, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 168, 28, 28), (131712, 1, 4704, 168))
        del arg74_1
        # Topologically Sorted Source Nodes: [conv2d_179], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(reinterpret_tensor(buf54, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 168, 28, 28), (131712, 1, 4704, 168))
        del arg75_1
        buf58 = empty_strided_cuda((8, 336, 28, 28), (263424, 784, 28, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_246, x_247, x_248], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_28.run(buf55, buf56, arg76_1, arg77_1, arg78_1, arg79_1, buf58, 2107392, grid=grid(2107392), stream=stream0)
        del arg76_1
        del arg77_1
        del arg78_1
        del arg79_1
        del buf55
        buf59 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [conv2d_180], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf58, buf59, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_180], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg80_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf60, (8, 168, 28, 28), (131712, 1, 4704, 168))
        del arg80_1
        buf61 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [conv2d_181], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf58, buf61, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_181], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg81_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf62, (8, 168, 28, 28), (131712, 1, 4704, 168))
        del arg81_1
        del buf61
        buf63 = buf58; del buf58  # reuse
        buf65 = empty_strided_cuda((8, 336, 1, 1), (336, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_249, x_250, x_251, x_se_68], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_31.run(buf60, buf62, arg82_1, arg83_1, arg84_1, arg85_1, buf63, buf65, 2688, 784, grid=grid(2688), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        del buf60
        # Topologically Sorted Source Nodes: [x_251, x_se_68, x_se_69], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg86_1
        del buf65
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_251, x_se_68, x_se_69, x_se_70], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_32.run(buf67, arg87_1, 224, grid=grid(224), stream=stream0)
        del arg87_1
        # Topologically Sorted Source Nodes: [x_251, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg88_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 336, 1, 1), (336, 1, 1, 1))
        del arg88_1
        del buf67
        buf69 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [x_251, x_se_68, x_se_69, x_se_70, x_se_71, sigmoid_17, x_252], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_33.run(buf69, buf68, arg89_1, 2107392, grid=grid(2107392), stream=stream0)
        del arg89_1
        buf70 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [conv2d_184], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf69, buf70, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_184], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 28, 28, 28), (21952, 1, 784, 28))
        del arg90_1
        buf72 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [conv2d_185], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf69, buf72, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_185], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 28, 28, 28), (21952, 1, 784, 28))
        del arg91_1
        del buf72
        buf74 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_253, x_254, x_255], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_34.run(buf74, buf71, buf73, arg92_1, arg93_1, arg94_1, arg95_1, 351232, grid=grid(351232), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf71
        del buf73
        # Topologically Sorted Source Nodes: [conv2d_186], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(reinterpret_tensor(buf74, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 168, 28, 28), (131712, 1, 4704, 168))
        del arg96_1
        # Topologically Sorted Source Nodes: [conv2d_187], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(reinterpret_tensor(buf74, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 168, 28, 28), (131712, 1, 4704, 168))
        del arg97_1
        buf78 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_256, x_257, x_258], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_28.run(buf75, buf76, arg98_1, arg99_1, arg100_1, arg101_1, buf78, 2107392, grid=grid(2107392), stream=stream0)
        del arg100_1
        del arg101_1
        del arg98_1
        del arg99_1
        del buf75
        buf79 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [conv2d_188], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf78, buf79, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_188], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg102_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf80, (8, 168, 28, 28), (131712, 1, 4704, 168))
        del arg102_1
        buf81 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [conv2d_189], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf78, buf81, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_189], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg103_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf82, (8, 168, 28, 28), (131712, 1, 4704, 168))
        del arg103_1
        del buf81
        buf83 = buf78; del buf78  # reuse
        buf85 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_259, x_260, x_261, x_se_72], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_31.run(buf80, buf82, arg104_1, arg105_1, arg106_1, arg107_1, buf83, buf85, 2688, 784, grid=grid(2688), stream=stream0)
        del arg104_1
        del arg105_1
        del arg106_1
        del arg107_1
        del buf80
        # Topologically Sorted Source Nodes: [x_261, x_se_72, x_se_73], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg108_1
        del buf85
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_261, x_se_72, x_se_73, x_se_74], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_32.run(buf87, arg109_1, 224, grid=grid(224), stream=stream0)
        del arg109_1
        # Topologically Sorted Source Nodes: [x_261, x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 336, 1, 1), (336, 1, 1, 1))
        del arg110_1
        del buf87
        buf89 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_261, x_se_72, x_se_73, x_se_74, x_se_75, sigmoid_18, x_262], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_33.run(buf89, buf88, arg111_1, 2107392, grid=grid(2107392), stream=stream0)
        del arg111_1
        buf90 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [conv2d_192], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf89, buf90, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_192], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 28, 28, 28), (21952, 1, 784, 28))
        del arg112_1
        buf92 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [conv2d_193], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf89, buf92, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_193], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg113_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 28, 28, 28), (21952, 1, 784, 28))
        del arg113_1
        del buf92
        buf94 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_263, x_264, x_265], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_34.run(buf94, buf91, buf93, arg114_1, arg115_1, arg116_1, arg117_1, 351232, grid=grid(351232), stream=stream0)
        del arg114_1
        del arg115_1
        del arg116_1
        del arg117_1
        del buf91
        del buf93
        # Topologically Sorted Source Nodes: [conv2d_194], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(reinterpret_tensor(buf94, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), arg118_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 168, 28, 28), (131712, 1, 4704, 168))
        del arg118_1
        # Topologically Sorted Source Nodes: [conv2d_195], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(reinterpret_tensor(buf94, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 168, 28, 28), (131712, 1, 4704, 168))
        del arg119_1
        buf98 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [x_266, x_267, x_268], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_28.run(buf95, buf96, arg120_1, arg121_1, arg122_1, arg123_1, buf98, 2107392, grid=grid(2107392), stream=stream0)
        del arg120_1
        del arg121_1
        del arg122_1
        del arg123_1
        del buf95
        buf99 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [conv2d_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf98, buf99, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_196], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg124_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf100, (8, 168, 28, 28), (131712, 1, 4704, 168))
        del arg124_1
        buf101 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [conv2d_197], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf98, buf101, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_197], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg125_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf102, (8, 168, 28, 28), (131712, 1, 4704, 168))
        del arg125_1
        del buf101
        buf103 = buf98; del buf98  # reuse
        buf105 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_269, x_270, x_271, x_se_76], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_31.run(buf100, buf102, arg126_1, arg127_1, arg128_1, arg129_1, buf103, buf105, 2688, 784, grid=grid(2688), stream=stream0)
        del arg126_1
        del arg127_1
        del arg128_1
        del arg129_1
        del buf100
        # Topologically Sorted Source Nodes: [x_271, x_se_76, x_se_77], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg130_1
        del buf105
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_271, x_se_76, x_se_77, x_se_78], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_32.run(buf107, arg131_1, 224, grid=grid(224), stream=stream0)
        del arg131_1
        # Topologically Sorted Source Nodes: [x_271, x_se_76, x_se_77, x_se_78, x_se_79], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 336, 1, 1), (336, 1, 1, 1))
        del arg132_1
        del buf107
        buf109 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [x_271, x_se_76, x_se_77, x_se_78, x_se_79, sigmoid_19, x_272], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_33.run(buf109, buf108, arg133_1, 2107392, grid=grid(2107392), stream=stream0)
        del arg133_1
        buf110 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [conv2d_200], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf109, buf110, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_200], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 28, 28, 28), (21952, 1, 784, 28))
        del arg134_1
        buf112 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [conv2d_201], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf109, buf112, 1344, 784, grid=grid(1344, 784), stream=stream0)
        del buf109
        # Topologically Sorted Source Nodes: [conv2d_201], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 28, 28, 28), (21952, 1, 784, 28))
        del arg135_1
        del buf112
        buf114 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_273, x_274, x_275], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_34.run(buf114, buf111, buf113, arg136_1, arg137_1, arg138_1, arg139_1, 351232, grid=grid(351232), stream=stream0)
        del arg136_1
        del arg137_1
        del arg138_1
        del arg139_1
        del buf111
        del buf113
        # Topologically Sorted Source Nodes: [x_273, x_274, x_275, x_276], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf115 = extern_kernels.convolution(buf114, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 336, 28, 28), (263424, 1, 9408, 336))
        del arg140_1
        del buf114
        buf116 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_277], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_35.run(buf116, arg141_1, arg142_1, arg143_1, arg144_1, 2107392, grid=grid(2107392), stream=stream0)
        del arg141_1
        del arg142_1
        del arg143_1
        del arg144_1
        buf117 = empty_strided_cuda((8, 112, 29, 29), (94192, 1, 3248, 112), torch.float32)
        # Topologically Sorted Source Nodes: [x_279], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_36.run(buf116, buf117, 753536, grid=grid(753536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_279, conv2d_203], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg145_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf118, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg145_1
        del buf117
        buf119 = empty_strided_cuda((8, 112, 31, 31), (107632, 1, 3472, 112), torch.float32)
        # Topologically Sorted Source Nodes: [x_280], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_37.run(buf116, buf119, 861056, grid=grid(861056), stream=stream0)
        # Topologically Sorted Source Nodes: [x_280, conv2d_204], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg146_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf120, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg146_1
        del buf119
        buf121 = empty_strided_cuda((8, 112, 33, 33), (121968, 1, 3696, 112), torch.float32)
        # Topologically Sorted Source Nodes: [x_281], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_38.run(buf116, buf121, 975744, grid=grid(975744), stream=stream0)
        del buf116
        # Topologically Sorted Source Nodes: [x_281, conv2d_205], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf122 = extern_kernels.convolution(buf121, arg147_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf122, (8, 112, 14, 14), (21952, 1, 1568, 112))
        del arg147_1
        del buf121
        buf123 = empty_strided_cuda((8, 336, 14, 14), (65856, 196, 14, 1), torch.float32)
        buf125 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_282, x_283, x_284, x_se_80], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_39.run(buf118, buf120, buf122, arg148_1, arg149_1, arg150_1, arg151_1, buf123, buf125, 2688, 196, grid=grid(2688), stream=stream0)
        del arg148_1
        del arg149_1
        del arg150_1
        del buf118
        del buf120
        del buf122
        # Topologically Sorted Source Nodes: [x_283, x_284, x_se_80, x_se_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf126 = extern_kernels.convolution(buf125, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 14, 1, 1), (14, 1, 1, 1))
        del arg152_1
        del buf125
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_283, x_284, x_se_80, x_se_81, x_se_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_40.run(buf127, arg153_1, 112, grid=grid(112), stream=stream0)
        del arg153_1
        # Topologically Sorted Source Nodes: [x_283, x_284, x_se_80, x_se_81, x_se_82, x_se_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 336, 1, 1), (336, 1, 1, 1))
        del arg154_1
        del buf127
        buf129 = empty_strided_cuda((8, 336, 14, 14), (65856, 1, 4704, 336), torch.float32)
        # Topologically Sorted Source Nodes: [x_283, x_284, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_41.run(buf123, arg151_1, buf128, arg155_1, buf129, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg151_1
        del arg155_1
        del buf123
        del buf128
        # Topologically Sorted Source Nodes: [x_283, x_284, x_se_80, x_se_81, x_se_82, x_se_83, sigmoid_20, x_285, x_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf130 = extern_kernels.convolution(buf129, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del arg156_1
        del buf129
        buf131 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_287], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_42.run(buf131, arg157_1, arg158_1, arg159_1, arg160_1, 163072, grid=grid(163072), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        # Topologically Sorted Source Nodes: [conv2d_209], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(reinterpret_tensor(buf131, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 312, 14, 14), (61152, 1, 4368, 312))
        del arg161_1
        # Topologically Sorted Source Nodes: [conv2d_210], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(reinterpret_tensor(buf131, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 312, 14, 14), (61152, 1, 4368, 312))
        del arg162_1
        buf135 = empty_strided_cuda((8, 624, 14, 14), (122304, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_288, x_289, x_290], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_43.run(buf132, buf133, arg163_1, arg164_1, arg165_1, arg166_1, buf135, 978432, grid=grid(978432), stream=stream0)
        del arg163_1
        del arg164_1
        del arg165_1
        del arg166_1
        del buf132
        buf136 = empty_strided_cuda((8, 156, 14, 14), (30576, 1, 2184, 156), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_211], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf135, buf136, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_211], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, arg167_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf137, (8, 156, 14, 14), (30576, 1, 2184, 156))
        del arg167_1
        buf138 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [conv2d_212], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf135, buf138, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_212], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, arg168_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf139, (8, 156, 14, 14), (30576, 1, 2184, 156))
        del arg168_1
        buf140 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [conv2d_213], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_46.run(buf135, buf140, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_213], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, arg169_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf141, (8, 156, 14, 14), (30576, 1, 2184, 156))
        del arg169_1
        buf142 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [conv2d_214], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf135, buf142, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_214], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, arg170_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf143, (8, 156, 14, 14), (30576, 1, 2184, 156))
        del arg170_1
        del buf142
        buf144 = buf135; del buf135  # reuse
        buf146 = empty_strided_cuda((8, 624, 1, 1), (624, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_291, x_292, x_293, x_se_84], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_48.run(buf137, buf139, buf141, buf143, arg171_1, arg172_1, arg173_1, arg174_1, buf144, buf146, 4992, 196, grid=grid(4992), stream=stream0)
        del arg171_1
        del arg172_1
        del buf137
        del buf139
        del buf141
        # Topologically Sorted Source Nodes: [x_292, x_293, x_se_84, x_se_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf147 = extern_kernels.convolution(buf146, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (8, 26, 1, 1), (26, 1, 1, 1))
        del arg175_1
        del buf146
        buf148 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_292, x_293, x_se_84, x_se_85, x_se_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_49.run(buf148, arg176_1, 208, grid=grid(208), stream=stream0)
        del arg176_1
        # Topologically Sorted Source Nodes: [x_292, x_293, x_se_84, x_se_85, x_se_86, x_se_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf149 = extern_kernels.convolution(buf148, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 624, 1, 1), (624, 1, 1, 1))
        del arg177_1
        del buf148
        buf150 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_292, x_293, x_se_84, x_se_85, x_se_86, x_se_87, sigmoid_21, x_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_50.run(buf150, arg173_1, arg174_1, buf149, arg178_1, 978432, grid=grid(978432), stream=stream0)
        del arg173_1
        del arg174_1
        del arg178_1
        buf151 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [conv2d_217], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf150, buf151, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_217], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 52, 14, 14), (10192, 1, 728, 52))
        del arg179_1
        buf153 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [conv2d_218], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf150, buf153, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_218], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 52, 14, 14), (10192, 1, 728, 52))
        del arg180_1
        del buf153
        buf155 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_295, x_296, x_297], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_53.run(buf155, buf152, buf154, arg181_1, arg182_1, arg183_1, arg184_1, 163072, grid=grid(163072), stream=stream0)
        del arg181_1
        del arg182_1
        del arg183_1
        del arg184_1
        del buf152
        del buf154
        # Topologically Sorted Source Nodes: [conv2d_219], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(reinterpret_tensor(buf155, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 312, 14, 14), (61152, 1, 4368, 312))
        del arg185_1
        # Topologically Sorted Source Nodes: [conv2d_220], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(reinterpret_tensor(buf155, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 312, 14, 14), (61152, 1, 4368, 312))
        del arg186_1
        buf159 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [x_298, x_299, x_300], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_43.run(buf156, buf157, arg187_1, arg188_1, arg189_1, arg190_1, buf159, 978432, grid=grid(978432), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        del buf156
        buf160 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [conv2d_221], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf159, buf160, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_221], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, arg191_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf161, (8, 156, 14, 14), (30576, 1, 2184, 156))
        del arg191_1
        buf162 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [conv2d_222], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf159, buf162, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_222], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, arg192_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf163, (8, 156, 14, 14), (30576, 1, 2184, 156))
        del arg192_1
        buf164 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [conv2d_223], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_46.run(buf159, buf164, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_223], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, arg193_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf165, (8, 156, 14, 14), (30576, 1, 2184, 156))
        del arg193_1
        buf166 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [conv2d_224], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf159, buf166, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_224], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, arg194_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf167, (8, 156, 14, 14), (30576, 1, 2184, 156))
        del arg194_1
        del buf166
        buf168 = buf159; del buf159  # reuse
        buf170 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [x_301, x_302, x_303, x_se_88], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_48.run(buf161, buf163, buf165, buf167, arg195_1, arg196_1, arg197_1, arg198_1, buf168, buf170, 4992, 196, grid=grid(4992), stream=stream0)
        del arg195_1
        del arg196_1
        del buf161
        del buf163
        del buf165
        # Topologically Sorted Source Nodes: [x_302, x_303, x_se_88, x_se_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf171 = extern_kernels.convolution(buf170, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 26, 1, 1), (26, 1, 1, 1))
        del arg199_1
        del buf170
        buf172 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_302, x_303, x_se_88, x_se_89, x_se_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_49.run(buf172, arg200_1, 208, grid=grid(208), stream=stream0)
        del arg200_1
        # Topologically Sorted Source Nodes: [x_302, x_303, x_se_88, x_se_89, x_se_90, x_se_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf173 = extern_kernels.convolution(buf172, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 624, 1, 1), (624, 1, 1, 1))
        del arg201_1
        del buf172
        buf174 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_302, x_303, x_se_88, x_se_89, x_se_90, x_se_91, sigmoid_22, x_304], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_50.run(buf174, arg197_1, arg198_1, buf173, arg202_1, 978432, grid=grid(978432), stream=stream0)
        del arg197_1
        del arg198_1
        del arg202_1
        buf175 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [conv2d_227], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf174, buf175, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_227], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 52, 14, 14), (10192, 1, 728, 52))
        del arg203_1
        buf177 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [conv2d_228], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf174, buf177, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_228], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 52, 14, 14), (10192, 1, 728, 52))
        del arg204_1
        del buf177
        buf179 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_305, x_306, x_307], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_53.run(buf179, buf176, buf178, arg205_1, arg206_1, arg207_1, arg208_1, 163072, grid=grid(163072), stream=stream0)
        del arg205_1
        del arg206_1
        del arg207_1
        del arg208_1
        del buf176
        del buf178
        # Topologically Sorted Source Nodes: [conv2d_229], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(reinterpret_tensor(buf179, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 312, 14, 14), (61152, 1, 4368, 312))
        del arg209_1
        # Topologically Sorted Source Nodes: [conv2d_230], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(reinterpret_tensor(buf179, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 312, 14, 14), (61152, 1, 4368, 312))
        del arg210_1
        buf183 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [x_308, x_309, x_310], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_43.run(buf180, buf181, arg211_1, arg212_1, arg213_1, arg214_1, buf183, 978432, grid=grid(978432), stream=stream0)
        del arg211_1
        del arg212_1
        del arg213_1
        del arg214_1
        del buf180
        buf184 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [conv2d_231], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf183, buf184, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_231], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, arg215_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf185, (8, 156, 14, 14), (30576, 1, 2184, 156))
        del arg215_1
        buf186 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [conv2d_232], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf183, buf186, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_232], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, arg216_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf187, (8, 156, 14, 14), (30576, 1, 2184, 156))
        del arg216_1
        buf188 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [conv2d_233], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_46.run(buf183, buf188, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_233], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, arg217_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf189, (8, 156, 14, 14), (30576, 1, 2184, 156))
        del arg217_1
        buf190 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [conv2d_234], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf183, buf190, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_234], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, arg218_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf191, (8, 156, 14, 14), (30576, 1, 2184, 156))
        del arg218_1
        del buf190
        buf192 = buf183; del buf183  # reuse
        buf194 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [x_311, x_312, x_313, x_se_92], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_48.run(buf185, buf187, buf189, buf191, arg219_1, arg220_1, arg221_1, arg222_1, buf192, buf194, 4992, 196, grid=grid(4992), stream=stream0)
        del arg219_1
        del arg220_1
        del buf185
        del buf187
        del buf189
        del buf191
        # Topologically Sorted Source Nodes: [x_312, x_313, x_se_92, x_se_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf195 = extern_kernels.convolution(buf194, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 26, 1, 1), (26, 1, 1, 1))
        del arg223_1
        del buf194
        buf196 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_312, x_313, x_se_92, x_se_93, x_se_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_49.run(buf196, arg224_1, 208, grid=grid(208), stream=stream0)
        del arg224_1
        # Topologically Sorted Source Nodes: [x_312, x_313, x_se_92, x_se_93, x_se_94, x_se_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf197 = extern_kernels.convolution(buf196, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 624, 1, 1), (624, 1, 1, 1))
        del arg225_1
        del buf196
        buf198 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [x_312, x_313, x_se_92, x_se_93, x_se_94, x_se_95, sigmoid_23, x_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_50.run(buf198, arg221_1, arg222_1, buf197, arg226_1, 978432, grid=grid(978432), stream=stream0)
        del arg221_1
        del arg222_1
        del arg226_1
        buf199 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [conv2d_237], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf198, buf199, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_237], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 52, 14, 14), (10192, 1, 728, 52))
        del arg227_1
        buf201 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [conv2d_238], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf198, buf201, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_238], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 52, 14, 14), (10192, 1, 728, 52))
        del arg228_1
        del buf201
        buf203 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_315, x_316, x_317], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_53.run(buf203, buf200, buf202, arg229_1, arg230_1, arg231_1, arg232_1, 163072, grid=grid(163072), stream=stream0)
        del arg229_1
        del arg230_1
        del arg231_1
        del arg232_1
        del buf200
        del buf202
        # Topologically Sorted Source Nodes: [x_315, x_316, x_317, x_318], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf204 = extern_kernels.convolution(buf203, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 624, 14, 14), (122304, 1, 8736, 624))
        del arg233_1
        del buf203
        buf205 = buf204; del buf204  # reuse
        buf206 = reinterpret_tensor(buf198, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [x_319, x_320], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_54.run(buf205, arg234_1, arg235_1, arg236_1, arg237_1, buf206, 978432, grid=grid(978432), stream=stream0)
        del arg234_1
        del arg235_1
        del arg236_1
        del arg237_1
        del buf205
        # Topologically Sorted Source Nodes: [x_320, x_321], Original ATen: [aten.silu, aten.convolution]
        buf207 = extern_kernels.convolution(buf206, arg238_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=624, bias=None)
        assert_size_stride(buf207, (8, 624, 14, 14), (122304, 1, 8736, 624))
        del arg238_1
        del buf206
        buf208 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_322], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_55.run(buf208, arg239_1, arg240_1, arg241_1, arg242_1, 978432, grid=grid(978432), stream=stream0)
        del arg239_1
        del arg240_1
        del arg241_1
        del arg242_1
        buf209 = empty_strided_cuda((8, 624, 1, 1, 2), (1248, 1, 9984, 9984, 624), torch.float32)
        # Topologically Sorted Source Nodes: [x_323, x_se_96], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_56.run(buf208, buf209, 9984, 98, grid=grid(9984), stream=stream0)
        buf211 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [x_323, x_se_96], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_57.run(buf209, buf211, 4992, 2, grid=grid(4992), stream=stream0)
        del buf209
        # Topologically Sorted Source Nodes: [x_323, x_se_96, x_se_97], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf212 = extern_kernels.convolution(buf211, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (8, 52, 1, 1), (52, 1, 1, 1))
        del arg243_1
        del buf211
        buf213 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [x_323, x_se_96, x_se_97, x_se_98], Original ATen: [aten.silu, aten.mean, aten.convolution]
        triton_poi_fused_convolution_mean_silu_58.run(buf213, arg244_1, 416, grid=grid(416), stream=stream0)
        del arg244_1
        # Topologically Sorted Source Nodes: [x_323, x_se_96, x_se_97, x_se_98, x_se_99], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf214 = extern_kernels.convolution(buf213, arg245_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 624, 1, 1), (624, 1, 1, 1))
        del arg245_1
        del buf213
        buf215 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [x_323, x_se_96, x_se_97, x_se_98, x_se_99, sigmoid_24, x_324], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_59.run(buf215, buf214, arg246_1, 978432, grid=grid(978432), stream=stream0)
        del arg246_1
        del buf214
        # Topologically Sorted Source Nodes: [x_323, x_se_96, x_se_97, x_se_98, x_se_99, sigmoid_24, x_324, x_325], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf216 = extern_kernels.convolution(buf215, arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 160, 14, 14), (31360, 1, 2240, 160))
        del arg247_1
        del buf215
        buf217 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [x_326], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_60.run(buf217, arg248_1, arg249_1, arg250_1, arg251_1, 250880, grid=grid(250880), stream=stream0)
        del arg248_1
        del arg249_1
        del arg250_1
        del arg251_1
        # Topologically Sorted Source Nodes: [conv2d_244], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(reinterpret_tensor(buf217, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg252_1
        # Topologically Sorted Source Nodes: [conv2d_245], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(reinterpret_tensor(buf217, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), arg253_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg253_1
        buf221 = empty_strided_cuda((8, 480, 14, 14), (94080, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_327, x_328, x_329], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_61.run(buf218, buf219, arg254_1, arg255_1, arg256_1, arg257_1, buf221, 752640, grid=grid(752640), stream=stream0)
        del arg254_1
        del arg255_1
        del arg256_1
        del arg257_1
        del buf218
        buf222 = empty_strided_cuda((8, 120, 14, 14), (23520, 1, 1680, 120), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_246], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf221, buf222, 960, 196, grid=grid(960, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_246], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, arg258_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf223, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg258_1
        buf224 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [conv2d_247], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf221, buf224, 960, 196, grid=grid(960, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_247], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, arg259_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf225, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg259_1
        buf226 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [conv2d_248], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf221, buf226, 960, 196, grid=grid(960, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_248], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, arg260_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf227, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg260_1
        buf228 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [conv2d_249], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf221, buf228, 960, 196, grid=grid(960, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_249], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, arg261_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf229, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg261_1
        del buf228
        buf230 = buf221; del buf221  # reuse
        buf232 = empty_strided_cuda((8, 480, 1, 1), (480, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_330, x_331, x_332, x_se_100], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_66.run(buf223, buf225, buf227, buf229, arg262_1, arg263_1, arg264_1, arg265_1, buf230, buf232, 3840, 196, grid=grid(3840), stream=stream0)
        del arg262_1
        del arg263_1
        del buf223
        del buf225
        del buf227
        # Topologically Sorted Source Nodes: [x_331, x_332, x_se_100, x_se_101], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf233 = extern_kernels.convolution(buf232, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (8, 80, 1, 1), (80, 1, 1, 1))
        del arg266_1
        del buf232
        buf234 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_331, x_332, x_se_100, x_se_101, x_se_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_67.run(buf234, arg267_1, 640, grid=grid(640), stream=stream0)
        del arg267_1
        # Topologically Sorted Source Nodes: [x_331, x_332, x_se_100, x_se_101, x_se_102, x_se_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf235 = extern_kernels.convolution(buf234, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg268_1
        del buf234
        buf236 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [x_331, x_332, x_se_100, x_se_101, x_se_102, x_se_103, sigmoid_25, x_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_68.run(buf236, arg264_1, arg265_1, buf235, arg269_1, 752640, grid=grid(752640), stream=stream0)
        del arg264_1
        del arg265_1
        del arg269_1
        buf237 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [conv2d_252], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf236, buf237, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_252], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg270_1
        buf239 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [conv2d_253], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_70.run(buf236, buf239, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_253], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg271_1
        del buf239
        buf241 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_334, x_335, x_336], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_71.run(buf241, buf238, buf240, arg272_1, arg273_1, arg274_1, arg275_1, 250880, grid=grid(250880), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        del buf238
        del buf240
        # Topologically Sorted Source Nodes: [conv2d_254], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(reinterpret_tensor(buf241, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg276_1
        # Topologically Sorted Source Nodes: [conv2d_255], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(reinterpret_tensor(buf241, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg277_1
        buf245 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [x_337, x_338, x_339], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_61.run(buf242, buf243, arg278_1, arg279_1, arg280_1, arg281_1, buf245, 752640, grid=grid(752640), stream=stream0)
        del arg278_1
        del arg279_1
        del arg280_1
        del arg281_1
        del buf242
        buf246 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [conv2d_256], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf245, buf246, 960, 196, grid=grid(960, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_256], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, arg282_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf247, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg282_1
        buf248 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [conv2d_257], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf245, buf248, 960, 196, grid=grid(960, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_257], Original ATen: [aten.convolution]
        buf249 = extern_kernels.convolution(buf248, arg283_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf249, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg283_1
        buf250 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [conv2d_258], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf245, buf250, 960, 196, grid=grid(960, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_258], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, arg284_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf251, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg284_1
        buf252 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [conv2d_259], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf245, buf252, 960, 196, grid=grid(960, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_259], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, arg285_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf253, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg285_1
        del buf252
        buf254 = buf245; del buf245  # reuse
        buf256 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [x_340, x_341, x_342, x_se_104], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_66.run(buf247, buf249, buf251, buf253, arg286_1, arg287_1, arg288_1, arg289_1, buf254, buf256, 3840, 196, grid=grid(3840), stream=stream0)
        del arg286_1
        del arg287_1
        del buf247
        del buf249
        del buf251
        # Topologically Sorted Source Nodes: [x_341, x_342, x_se_104, x_se_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf257 = extern_kernels.convolution(buf256, arg290_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (8, 80, 1, 1), (80, 1, 1, 1))
        del arg290_1
        del buf256
        buf258 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [x_341, x_342, x_se_104, x_se_105, x_se_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_67.run(buf258, arg291_1, 640, grid=grid(640), stream=stream0)
        del arg291_1
        # Topologically Sorted Source Nodes: [x_341, x_342, x_se_104, x_se_105, x_se_106, x_se_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf259 = extern_kernels.convolution(buf258, arg292_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg292_1
        del buf258
        buf260 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [x_341, x_342, x_se_104, x_se_105, x_se_106, x_se_107, sigmoid_26, x_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_68.run(buf260, arg288_1, arg289_1, buf259, arg293_1, 752640, grid=grid(752640), stream=stream0)
        del arg288_1
        del arg289_1
        del arg293_1
        buf261 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [conv2d_262], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf260, buf261, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_262], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg294_1
        buf263 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [conv2d_263], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_70.run(buf260, buf263, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_263], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, arg295_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg295_1
        del buf263
        buf265 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [x_344, x_345, x_346], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_71.run(buf265, buf262, buf264, arg296_1, arg297_1, arg298_1, arg299_1, 250880, grid=grid(250880), stream=stream0)
        del arg296_1
        del arg297_1
        del arg298_1
        del arg299_1
        del buf262
        del buf264
        # Topologically Sorted Source Nodes: [conv2d_264], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(reinterpret_tensor(buf265, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg300_1
        # Topologically Sorted Source Nodes: [conv2d_265], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(reinterpret_tensor(buf265, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (8, 240, 14, 14), (47040, 1, 3360, 240))
        del arg301_1
        buf269 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [x_347, x_348, x_349], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_silu_61.run(buf266, buf267, arg302_1, arg303_1, arg304_1, arg305_1, buf269, 752640, grid=grid(752640), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        buf270 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [conv2d_266], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf269, buf270, 960, 196, grid=grid(960, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_266], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, arg306_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf271, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg306_1
        buf272 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [conv2d_267], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf269, buf272, 960, 196, grid=grid(960, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_267], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, arg307_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf273, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg307_1
        buf274 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [conv2d_268], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf269, buf274, 960, 196, grid=grid(960, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_268], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, arg308_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf275, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg308_1
        buf276 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [conv2d_269], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf269, buf276, 960, 196, grid=grid(960, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_269], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, arg309_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf277, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg309_1
        del buf276
        buf278 = buf269; del buf269  # reuse
        buf280 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [x_350, x_351, x_352, x_se_108], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_66.run(buf271, buf273, buf275, buf277, arg310_1, arg311_1, arg312_1, arg313_1, buf278, buf280, 3840, 196, grid=grid(3840), stream=stream0)
        del arg310_1
        del arg311_1
        del buf271
        del buf273
        del buf275
        del buf277
        # Topologically Sorted Source Nodes: [x_351, x_352, x_se_108, x_se_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf281 = extern_kernels.convolution(buf280, arg314_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (8, 80, 1, 1), (80, 1, 1, 1))
        del arg314_1
        del buf280
        buf282 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [x_351, x_352, x_se_108, x_se_109, x_se_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_67.run(buf282, arg315_1, 640, grid=grid(640), stream=stream0)
        del arg315_1
        # Topologically Sorted Source Nodes: [x_351, x_352, x_se_108, x_se_109, x_se_110, x_se_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf283 = extern_kernels.convolution(buf282, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg316_1
        del buf282
        buf284 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [x_351, x_352, x_se_108, x_se_109, x_se_110, x_se_111, sigmoid_27, x_353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_68.run(buf284, arg312_1, arg313_1, buf283, arg317_1, 752640, grid=grid(752640), stream=stream0)
        del arg312_1
        del arg313_1
        del arg317_1
        del buf283
        buf285 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [conv2d_272], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf284, buf285, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_272], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, arg318_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg318_1
        buf287 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [conv2d_273], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_70.run(buf284, buf287, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del buf284
        # Topologically Sorted Source Nodes: [conv2d_273], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, arg319_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg319_1
        buf289 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [x_354, x_355, x_356], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_71.run(buf289, buf286, buf288, arg320_1, arg321_1, arg322_1, arg323_1, 250880, grid=grid(250880), stream=stream0)
        del arg320_1
        del arg321_1
        del arg322_1
        del arg323_1
        del buf286
        del buf288
        # Topologically Sorted Source Nodes: [x_354, x_355, x_356, x_357], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf290 = extern_kernels.convolution(buf289, arg324_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (8, 960, 14, 14), (188160, 1, 13440, 960))
        del arg324_1
        del buf289
        buf291 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [x_358], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_72.run(buf291, arg325_1, arg326_1, arg327_1, arg328_1, 1505280, grid=grid(1505280), stream=stream0)
        del arg325_1
        del arg326_1
        del arg327_1
        del arg328_1
        buf292 = empty_strided_cuda((8, 240, 15, 15), (54000, 1, 3600, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_360], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_73.run(buf291, buf292, 432000, grid=grid(432000), stream=stream0)
        # Topologically Sorted Source Nodes: [x_360, conv2d_275], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf293 = extern_kernels.convolution(buf292, arg329_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf293, (8, 240, 7, 7), (11760, 1, 1680, 240))
        del arg329_1
        del buf292
        buf294 = empty_strided_cuda((8, 240, 17, 17), (69360, 1, 4080, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_361], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_74.run(buf291, buf294, 554880, grid=grid(554880), stream=stream0)
        # Topologically Sorted Source Nodes: [x_361, conv2d_276], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf295 = extern_kernels.convolution(buf294, arg330_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf295, (8, 240, 7, 7), (11760, 1, 1680, 240))
        del arg330_1
        del buf294
        buf296 = empty_strided_cuda((8, 240, 19, 19), (86640, 1, 4560, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_362], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_75.run(buf291, buf296, 693120, grid=grid(693120), stream=stream0)
        # Topologically Sorted Source Nodes: [x_362, conv2d_277], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf297 = extern_kernels.convolution(buf296, arg331_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf297, (8, 240, 7, 7), (11760, 1, 1680, 240))
        del arg331_1
        del buf296
        buf298 = empty_strided_cuda((8, 240, 21, 21), (105840, 1, 5040, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_363], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_76.run(buf291, buf298, 846720, grid=grid(846720), stream=stream0)
        del buf291
        # Topologically Sorted Source Nodes: [x_363, conv2d_278], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf299 = extern_kernels.convolution(buf298, arg332_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf299, (8, 240, 7, 7), (11760, 1, 1680, 240))
        del arg332_1
        del buf298
        buf300 = reinterpret_tensor(buf287, (8, 960, 7, 7), (47040, 49, 7, 1), 0); del buf287  # reuse
        buf302 = empty_strided_cuda((8, 960, 1, 1), (960, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_364, x_365, x_366, x_se_112], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_77.run(buf293, buf295, buf297, buf299, arg333_1, arg334_1, arg335_1, arg336_1, buf300, buf302, 7680, 49, grid=grid(7680), stream=stream0)
        del arg333_1
        del arg334_1
        del buf293
        del buf295
        del buf297
        del buf299
        # Topologically Sorted Source Nodes: [x_365, x_366, x_se_112, x_se_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf303 = extern_kernels.convolution(buf302, arg337_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (8, 80, 1, 1), (80, 1, 1, 1))
        del arg337_1
        del buf302
        buf304 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [x_365, x_366, x_se_112, x_se_113, x_se_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_67.run(buf304, arg338_1, 640, grid=grid(640), stream=stream0)
        del arg338_1
        # Topologically Sorted Source Nodes: [x_365, x_366, x_se_112, x_se_113, x_se_114, x_se_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf305 = extern_kernels.convolution(buf304, arg339_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf305, (8, 960, 1, 1), (960, 1, 1, 1))
        del arg339_1
        del buf304
        buf306 = reinterpret_tensor(buf266, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [x_365, x_366, x_se_112, x_se_113, x_se_114, x_se_115, sigmoid_28, x_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_78.run(buf300, arg335_1, arg336_1, buf305, arg340_1, buf306, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg335_1
        del arg336_1
        del arg340_1
        del buf300
        del buf305
        # Topologically Sorted Source Nodes: [x_365, x_366, x_se_112, x_se_113, x_se_114, x_se_115, sigmoid_28, x_367, x_368], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        buf307 = extern_kernels.convolution(buf306, arg341_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (8, 264, 7, 7), (12936, 1, 1848, 264))
        del arg341_1
        del buf306
        buf308 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [x_369], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_79.run(buf308, arg342_1, arg343_1, arg344_1, arg345_1, 103488, grid=grid(103488), stream=stream0)
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        # Topologically Sorted Source Nodes: [x_370], Original ATen: [aten.convolution]
        buf309 = extern_kernels.convolution(buf308, arg346_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
        del arg346_1
        buf310 = buf309; del buf309  # reuse
        buf311 = empty_strided_cuda((8, 1584, 7, 7), (77616, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_371, x_372], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_80.run(buf310, arg347_1, arg348_1, arg349_1, arg350_1, buf311, 392, 1584, grid=grid(392, 1584), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        del arg350_1
        del buf310
        buf312 = empty_strided_cuda((8, 396, 7, 7), (19404, 1, 2772, 396), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_283], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf311, buf312, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_283], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, arg351_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf313, (8, 396, 7, 7), (19404, 1, 2772, 396))
        del arg351_1
        buf314 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [conv2d_284], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_82.run(buf311, buf314, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_284], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(buf314, arg352_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf315, (8, 396, 7, 7), (19404, 1, 2772, 396))
        del arg352_1
        buf316 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [conv2d_285], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf311, buf316, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_285], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, arg353_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf317, (8, 396, 7, 7), (19404, 1, 2772, 396))
        del arg353_1
        buf318 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [conv2d_286], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_84.run(buf311, buf318, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_286], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, arg354_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf319, (8, 396, 7, 7), (19404, 1, 2772, 396))
        del arg354_1
        del buf318
        buf320 = empty_strided_cuda((8, 1584, 7, 7), (77632, 49, 7, 1), torch.float32)
        buf322 = empty_strided_cuda((8, 1584, 1, 1), (1584, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_373, x_374, x_375, x_se_116], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_85.run(buf313, buf315, buf317, buf319, arg355_1, arg356_1, arg357_1, arg358_1, buf320, buf322, 12672, 49, grid=grid(12672), stream=stream0)
        del arg355_1
        del arg356_1
        del buf313
        del buf315
        del buf317
        # Topologically Sorted Source Nodes: [x_374, x_375, x_se_116, x_se_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf323 = extern_kernels.convolution(buf322, arg359_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf323, (8, 132, 1, 1), (132, 1, 1, 1))
        del arg359_1
        del buf322
        buf324 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [x_374, x_375, x_se_116, x_se_117, x_se_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_86.run(buf324, arg360_1, 1056, grid=grid(1056), stream=stream0)
        del arg360_1
        # Topologically Sorted Source Nodes: [x_374, x_375, x_se_116, x_se_117, x_se_118, x_se_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf325 = extern_kernels.convolution(buf324, arg361_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (8, 1584, 1, 1), (1584, 1, 1, 1))
        del arg361_1
        del buf324
        buf326 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [x_374, x_375, x_se_116, x_se_117, x_se_118, x_se_119, sigmoid_29, x_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_87.run(buf320, arg357_1, arg358_1, buf325, arg362_1, buf326, 620928, grid=grid(620928), stream=stream0)
        del arg357_1
        del arg358_1
        del arg362_1
        buf327 = empty_strided_cuda((8, 792, 7, 7), (38808, 1, 5544, 792), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_289], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_88.run(buf326, buf327, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_289], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf327, arg363_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (8, 132, 7, 7), (6468, 1, 924, 132))
        del arg363_1
        buf329 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [conv2d_290], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf326, buf329, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_290], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, arg364_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (8, 132, 7, 7), (6468, 1, 924, 132))
        del arg364_1
        buf331 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [x_377, x_378, x_379], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_90.run(buf331, buf328, buf330, arg365_1, arg366_1, arg367_1, arg368_1, 103488, grid=grid(103488), stream=stream0)
        del arg365_1
        del arg366_1
        del arg367_1
        del arg368_1
        del buf328
        del buf330
        # Topologically Sorted Source Nodes: [x_380], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, arg369_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
        del arg369_1
        buf333 = buf332; del buf332  # reuse
        buf334 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [x_381, x_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_80.run(buf333, arg370_1, arg371_1, arg372_1, arg373_1, buf334, 392, 1584, grid=grid(392, 1584), stream=stream0)
        del arg370_1
        del arg371_1
        del arg372_1
        del arg373_1
        del buf333
        buf335 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [conv2d_292], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf334, buf335, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_292], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, arg374_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf336, (8, 396, 7, 7), (19404, 1, 2772, 396))
        del arg374_1
        buf337 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [conv2d_293], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_82.run(buf334, buf337, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_293], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf337, arg375_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf338, (8, 396, 7, 7), (19404, 1, 2772, 396))
        del arg375_1
        buf339 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [conv2d_294], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf334, buf339, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_294], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, arg376_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf340, (8, 396, 7, 7), (19404, 1, 2772, 396))
        del arg376_1
        buf341 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [conv2d_295], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_84.run(buf334, buf341, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_295], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, arg377_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf342, (8, 396, 7, 7), (19404, 1, 2772, 396))
        del arg377_1
        del buf341
        buf343 = buf320; del buf320  # reuse
        buf345 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [x_383, x_384, x_385, x_se_120], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_85.run(buf336, buf338, buf340, buf342, arg378_1, arg379_1, arg380_1, arg381_1, buf343, buf345, 12672, 49, grid=grid(12672), stream=stream0)
        del arg378_1
        del arg379_1
        del buf336
        del buf338
        del buf340
        # Topologically Sorted Source Nodes: [x_384, x_385, x_se_120, x_se_121], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf346 = extern_kernels.convolution(buf345, arg382_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (8, 132, 1, 1), (132, 1, 1, 1))
        del arg382_1
        del buf345
        buf347 = buf346; del buf346  # reuse
        # Topologically Sorted Source Nodes: [x_384, x_385, x_se_120, x_se_121, x_se_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_86.run(buf347, arg383_1, 1056, grid=grid(1056), stream=stream0)
        del arg383_1
        # Topologically Sorted Source Nodes: [x_384, x_385, x_se_120, x_se_121, x_se_122, x_se_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf348 = extern_kernels.convolution(buf347, arg384_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (8, 1584, 1, 1), (1584, 1, 1, 1))
        del arg384_1
        del buf347
        buf349 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [x_384, x_385, x_se_120, x_se_121, x_se_122, x_se_123, sigmoid_30, x_386], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_87.run(buf343, arg380_1, arg381_1, buf348, arg385_1, buf349, 620928, grid=grid(620928), stream=stream0)
        del arg380_1
        del arg381_1
        del arg385_1
        buf350 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [conv2d_298], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_88.run(buf349, buf350, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_298], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, arg386_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (8, 132, 7, 7), (6468, 1, 924, 132))
        del arg386_1
        buf352 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [conv2d_299], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf349, buf352, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_299], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, arg387_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (8, 132, 7, 7), (6468, 1, 924, 132))
        del arg387_1
        buf354 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [x_387, x_388, x_389], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_90.run(buf354, buf351, buf353, arg388_1, arg389_1, arg390_1, arg391_1, 103488, grid=grid(103488), stream=stream0)
        del arg388_1
        del arg389_1
        del arg390_1
        del arg391_1
        del buf351
        del buf353
        # Topologically Sorted Source Nodes: [x_390], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf354, arg392_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
        del arg392_1
        buf356 = buf355; del buf355  # reuse
        buf357 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [x_391, x_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_80.run(buf356, arg393_1, arg394_1, arg395_1, arg396_1, buf357, 392, 1584, grid=grid(392, 1584), stream=stream0)
        del arg393_1
        del arg394_1
        del arg395_1
        del arg396_1
        del buf356
        buf358 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [conv2d_301], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf357, buf358, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_301], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, arg397_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf359, (8, 396, 7, 7), (19404, 1, 2772, 396))
        del arg397_1
        buf360 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [conv2d_302], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_82.run(buf357, buf360, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_302], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, arg398_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf361, (8, 396, 7, 7), (19404, 1, 2772, 396))
        del arg398_1
        buf362 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [conv2d_303], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf357, buf362, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_303], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, arg399_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf363, (8, 396, 7, 7), (19404, 1, 2772, 396))
        del arg399_1
        buf364 = buf362; del buf362  # reuse
        # Topologically Sorted Source Nodes: [conv2d_304], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_84.run(buf357, buf364, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_304], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, arg400_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf365, (8, 396, 7, 7), (19404, 1, 2772, 396))
        del arg400_1
        del buf364
        buf366 = buf343; del buf343  # reuse
        buf368 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [x_393, x_394, x_395, x_se_124], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.silu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_85.run(buf359, buf361, buf363, buf365, arg401_1, arg402_1, arg403_1, arg404_1, buf366, buf368, 12672, 49, grid=grid(12672), stream=stream0)
        del arg401_1
        del arg402_1
        del buf359
        del buf361
        del buf363
        del buf365
        # Topologically Sorted Source Nodes: [x_394, x_395, x_se_124, x_se_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf369 = extern_kernels.convolution(buf368, arg405_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (8, 132, 1, 1), (132, 1, 1, 1))
        del arg405_1
        del buf368
        buf370 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [x_394, x_395, x_se_124, x_se_125, x_se_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_silu_86.run(buf370, arg406_1, 1056, grid=grid(1056), stream=stream0)
        del arg406_1
        # Topologically Sorted Source Nodes: [x_394, x_395, x_se_124, x_se_125, x_se_126, x_se_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution]
        buf371 = extern_kernels.convolution(buf370, arg407_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (8, 1584, 1, 1), (1584, 1, 1, 1))
        del arg407_1
        del buf370
        buf372 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [x_394, x_395, x_se_124, x_se_125, x_se_126, x_se_127, sigmoid_31, x_396], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu, aten.mean, aten.convolution, aten.sigmoid, aten.mul]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_mul_sigmoid_silu_87.run(buf366, arg403_1, arg404_1, buf371, arg408_1, buf372, 620928, grid=grid(620928), stream=stream0)
        del arg403_1
        del arg404_1
        del arg408_1
        del buf366
        del buf371
        buf373 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [conv2d_307], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_88.run(buf372, buf373, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_307], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, arg409_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (8, 132, 7, 7), (6468, 1, 924, 132))
        del arg409_1
        buf375 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [conv2d_308], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf372, buf375, 6336, 49, grid=grid(6336, 49), stream=stream0)
        del buf372
        # Topologically Sorted Source Nodes: [conv2d_308], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, arg410_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (8, 132, 7, 7), (6468, 1, 924, 132))
        del arg410_1
        del buf375
        buf377 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [x_397, x_398, x_399], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_90.run(buf377, buf374, buf376, arg411_1, arg412_1, arg413_1, arg414_1, 103488, grid=grid(103488), stream=stream0)
        del arg411_1
        del arg412_1
        del arg413_1
        del arg414_1
        del buf374
        del buf376
        # Topologically Sorted Source Nodes: [x_397, x_398, x_399, x_400], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf378 = extern_kernels.convolution(buf377, arg415_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf378, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
        del arg415_1
        del buf377
        buf380 = empty_strided_cuda((8, 1536, 1, 1), (1536, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [x_401, x_402, x_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_91.run(buf378, arg416_1, arg417_1, arg418_1, arg419_1, buf380, 12288, 49, grid=grid(12288), stream=stream0)
        del arg416_1
        del arg417_1
        del arg418_1
        del arg419_1
        del buf378
        buf381 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_405], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg421_1, reinterpret_tensor(buf380, (8, 1536), (1536, 1), 0), reinterpret_tensor(arg420_1, (1536, 1000), (1, 1536), 0), alpha=1, beta=1, out=buf381)
        del arg420_1
        del arg421_1
        del buf380
    return (buf381, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((60, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((60, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((60, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((20, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((56, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((336, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((112, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((112, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((14, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((336, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((104, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((624, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((624, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((52, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((624, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((160, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((240, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((240, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((960, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((264, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((1536, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((1000, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tf_mixnet_l', benchmark_compiled_module)
