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
# Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_88 => convolution_32
# Graph fragment:
#   %convolution_32 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/2l/c2lrz6pkojr7t6lps2wacgvgg7n2n6usbeh56sh2ynlaoplkpdp6.py
# Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_88 => convolution_32
# Graph fragment:
#   %convolution_32 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
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


# kernel path: /tmp/torchinductor_sahanp/ty/ctyamezdm5aetkqnjecaj7moupqkkwkt6vzczee3goqyoawpkfwo.py
# Topologically Sorted Source Nodes: [x_89, x_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_89 => add_85, mul_112, mul_113, sub_27
#   x_90 => add_86, clamp_max_30, clamp_min_30, div_30, mul_114
# Graph fragment:
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_32, %unsqueeze_217), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %unsqueeze_221), kwargs = {})
#   %add_85 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %unsqueeze_223), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_85, 3), kwargs = {})
#   %clamp_min_30 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_86, 0), kwargs = {})
#   %clamp_max_30 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_30, 6), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_85, %clamp_max_30), kwargs = {})
#   %div_30 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_114, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 8
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


# kernel path: /tmp/torchinductor_sahanp/rg/crgbscyclknsf34shdji72oa56kuw5hpal5fblwvqn5zkrmqcqz5.py
# Topologically Sorted Source Nodes: [x_95, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_95 => add_91, mul_120, mul_121, sub_29
#   x_96 => add_92, clamp_max_32, clamp_min_32, div_32, mul_122
# Graph fragment:
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_34, %unsqueeze_233), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_235), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_120, %unsqueeze_237), kwargs = {})
#   %add_91 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_121, %unsqueeze_239), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, 3), kwargs = {})
#   %clamp_min_32 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_92, 0), kwargs = {})
#   %clamp_max_32 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_32, 6), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_91, %clamp_max_32), kwargs = {})
#   %div_32 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_122, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/3c/c3ch273vx4mhjkpulinlftjqbvwngiwuamr2ypkligfoimjj4l2i.py
# Topologically Sorted Source Nodes: [x_98, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_98 => add_94, mul_124, mul_125, sub_30
#   x_99 => add_95, clamp_max_33, clamp_min_33, div_33, mul_126
# Graph fragment:
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_241), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %unsqueeze_245), kwargs = {})
#   %add_94 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %unsqueeze_247), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, 3), kwargs = {})
#   %clamp_min_33 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_95, 0), kwargs = {})
#   %clamp_max_33 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_33, 6), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_94, %clamp_max_33), kwargs = {})
#   %div_33 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_126, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
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


# kernel path: /tmp/torchinductor_sahanp/yr/cyrrcufacdnjezxycuk4xb2but7irhkznxikrvsvs7lufp365kpp.py
# Topologically Sorted Source Nodes: [x_101, x_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_101 => add_97, mul_128, mul_129, sub_31
#   x_102 => add_98, clamp_max_34, clamp_min_34, div_34, mul_130
# Graph fragment:
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_36, %unsqueeze_249), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_251), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_128, %unsqueeze_253), kwargs = {})
#   %add_97 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_129, %unsqueeze_255), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_97, 3), kwargs = {})
#   %clamp_min_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_98, 0), kwargs = {})
#   %clamp_max_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_34, 6), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_97, %clamp_max_34), kwargs = {})
#   %div_34 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_130, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
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


# kernel path: /tmp/torchinductor_sahanp/x4/cx42d3tulw3i2q6h7in4pnsia5kv45nf4ecfvb44yxzso5cwqtrz.py
# Topologically Sorted Source Nodes: [x_110, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_110 => add_106, mul_140, mul_141, sub_34
#   x_111 => add_107, clamp_max_37, clamp_min_37, div_37, mul_142
# Graph fragment:
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_273), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_275), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_140, %unsqueeze_277), kwargs = {})
#   %add_106 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_141, %unsqueeze_279), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_106, 3), kwargs = {})
#   %clamp_min_37 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_107, 0), kwargs = {})
#   %clamp_max_37 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_37, 6), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_106, %clamp_max_37), kwargs = {})
#   %div_37 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_142, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/xf/cxf5k6cv5nxaolhvvvfvmr4zgbrk7ytyt76mct5trbatthrv2lyn.py
# Topologically Sorted Source Nodes: [x_113, x_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_113 => add_109, mul_144, mul_145, sub_35
#   x_114 => add_110, clamp_max_38, clamp_min_38, div_38, mul_146
# Graph fragment:
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_40, %unsqueeze_281), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_283), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_144, %unsqueeze_285), kwargs = {})
#   %add_109 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_145, %unsqueeze_287), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_109, 3), kwargs = {})
#   %clamp_min_38 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_110, 0), kwargs = {})
#   %clamp_max_38 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_38, 6), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_109, %clamp_max_38), kwargs = {})
#   %div_38 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_146, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
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


# kernel path: /tmp/torchinductor_sahanp/hv/chvackfjtkdyjgqd2sn2gudynbjv3zcbvoydaqtselfildivkx3a.py
# Topologically Sorted Source Nodes: [x_122, x_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_122 => add_118, mul_156, mul_157, sub_38
#   x_123 => add_119, clamp_max_41, clamp_min_41, div_41, mul_158
# Graph fragment:
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_305), kwargs = {})
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_156, %unsqueeze_309), kwargs = {})
#   %add_118 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_157, %unsqueeze_311), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_118, 3), kwargs = {})
#   %clamp_min_41 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_119, 0), kwargs = {})
#   %clamp_max_41 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_41, 6), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_118, %clamp_max_41), kwargs = {})
#   %div_41 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_158, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/gk/cgkg7kd2butq2jah7fbpcn5fjon6m42d4g5n5372c3bvndjgy7y7.py
# Topologically Sorted Source Nodes: [x_125, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_125 => add_121, mul_160, mul_161, sub_39
#   x_126 => add_122, clamp_max_42, clamp_min_42, div_42, mul_162
# Graph fragment:
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_313), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_160, %unsqueeze_317), kwargs = {})
#   %add_121 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_161, %unsqueeze_319), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_121, 3), kwargs = {})
#   %clamp_min_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_122, 0), kwargs = {})
#   %clamp_max_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_42, 6), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_121, %clamp_max_42), kwargs = {})
#   %div_42 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_162, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
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


# kernel path: /tmp/torchinductor_sahanp/6n/c6nxvs55id3yrj5pgo5ojyfjo2ey5w3xeuirglu5mo24xdvdcuaz.py
# Topologically Sorted Source Nodes: [x_158], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_158 => add_154, mul_204, mul_205, sub_50
# Graph fragment:
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_55, %unsqueeze_401), kwargs = {})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_403), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_204, %unsqueeze_405), kwargs = {})
#   %add_154 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_205, %unsqueeze_407), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
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


# kernel path: /tmp/torchinductor_sahanp/3v/c3v7nrqhcvyvlrmutnptymcakqjkd7l7arhmdkdbcgpri5gfun5n.py
# Topologically Sorted Source Nodes: [x_159, x_se_8], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_159 => add_155, clamp_max_53, clamp_min_53, div_53, mul_206
#   x_se_8 => mean_3
# Graph fragment:
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_154, 3), kwargs = {})
#   %clamp_min_53 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_155, 0), kwargs = {})
#   %clamp_max_53 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_53, 6), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_154, %clamp_max_53), kwargs = {})
#   %div_53 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_206, 6), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_53, [2, 3], True), kwargs = {})
triton_per_fused_hardswish_mean_11 = async_compile.triton('triton_per_fused_hardswish_mean_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_11(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (6272*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/he/chemrr4xt755jzwdtmg5sqyn5yeaczwsvstf7vjoimju2klfiox5.py
# Topologically Sorted Source Nodes: [x_159, x_se_8, x_se_9, x_se_10], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_159 => add_155, clamp_max_53, clamp_min_53, div_53, mul_206
#   x_se_10 => relu_2
#   x_se_8 => mean_3
#   x_se_9 => convolution_56
# Graph fragment:
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_154, 3), kwargs = {})
#   %clamp_min_53 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_155, 0), kwargs = {})
#   %clamp_max_53 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_53, 6), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_154, %clamp_max_53), kwargs = {})
#   %div_53 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_206, 6), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_53, [2, 3], True), kwargs = {})
#   %convolution_56 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_3, %arg121_1, %arg122_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_56,), kwargs = {})
triton_poi_fused_convolution_hardswish_mean_relu_12 = async_compile.triton('triton_poi_fused_convolution_hardswish_mean_relu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardswish_mean_relu_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/d7/cd7aqdcssn3dznwbknik45o2p6doxysc24fjowthqb4qvr3womcf.py
# Topologically Sorted Source Nodes: [x_159, x_se_8, x_se_9, x_se_10, x_se_11, hardsigmoid_2, x_160], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_2 => add_156, clamp_max_54, clamp_min_54, div_54
#   x_159 => add_155, clamp_max_53, clamp_min_53, div_53, mul_206
#   x_160 => mul_207
#   x_se_10 => relu_2
#   x_se_11 => convolution_57
#   x_se_8 => mean_3
#   x_se_9 => convolution_56
# Graph fragment:
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_154, 3), kwargs = {})
#   %clamp_min_53 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_155, 0), kwargs = {})
#   %clamp_max_53 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_53, 6), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_154, %clamp_max_53), kwargs = {})
#   %div_53 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_206, 6), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_53, [2, 3], True), kwargs = {})
#   %convolution_56 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_3, %arg121_1, %arg122_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_56,), kwargs = {})
#   %convolution_57 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg123_1, %arg124_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_156 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_57, 3), kwargs = {})
#   %clamp_min_54 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_156, 0), kwargs = {})
#   %clamp_max_54 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_54, 6), kwargs = {})
#   %div_54 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_54, 6), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_53, %div_54), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_13 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_13(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x2 = (xindex // 6272)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + (128*x2)), xmask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_sahanp/i5/ci5eztvxmg7wyb2jf7x5vs2nbmh7pho6s7jpar2qrlp7kiuhdrqa.py
# Topologically Sorted Source Nodes: [x_162, x_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_162 => add_158, mul_209, mul_210, sub_51
#   x_163 => add_159, clamp_max_55, clamp_min_55, div_55, mul_211
# Graph fragment:
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_58, %unsqueeze_409), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_411), kwargs = {})
#   %mul_210 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_209, %unsqueeze_413), kwargs = {})
#   %add_158 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_210, %unsqueeze_415), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_158, 3), kwargs = {})
#   %clamp_min_55 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_159, 0), kwargs = {})
#   %clamp_max_55 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_55, 6), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_158, %clamp_max_55), kwargs = {})
#   %div_55 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_211, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
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


# kernel path: /tmp/torchinductor_sahanp/h4/ch4ddojsjveoadkg3oi2bpur6qg2frkehulv3632emw2zboxlagm.py
# Topologically Sorted Source Nodes: [x_165], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_165 => add_161, mul_213, mul_214, sub_52
# Graph fragment:
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_59, %unsqueeze_417), kwargs = {})
#   %mul_213 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_419), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_213, %unsqueeze_421), kwargs = {})
#   %add_161 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_214, %unsqueeze_423), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
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


# kernel path: /tmp/torchinductor_sahanp/oe/coekxrsssusxfeh7ymnvcflwng3r7qczpdrcv4az7z3oh7wfueeb.py
# Topologically Sorted Source Nodes: [x_166, x_se_12], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_166 => add_162, clamp_max_56, clamp_min_56, div_56, mul_215
#   x_se_12 => mean_4
# Graph fragment:
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_161, 3), kwargs = {})
#   %clamp_min_56 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_162, 0), kwargs = {})
#   %clamp_max_56 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_56, 6), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %clamp_max_56), kwargs = {})
#   %div_56 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_215, 6), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_56, [2, 3], True), kwargs = {})
triton_per_fused_hardswish_mean_16 = async_compile.triton('triton_per_fused_hardswish_mean_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_16(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (12544*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/fs/cfsmuvxdxzhocsivdnnhx7nqjsgiiyd2azganzeh7lcynwcpyapy.py
# Topologically Sorted Source Nodes: [x_166, x_se_12, x_se_13, x_se_14], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_166 => add_162, clamp_max_56, clamp_min_56, div_56, mul_215
#   x_se_12 => mean_4
#   x_se_13 => convolution_60
#   x_se_14 => relu_3
# Graph fragment:
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_161, 3), kwargs = {})
#   %clamp_min_56 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_162, 0), kwargs = {})
#   %clamp_max_56 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_56, 6), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %clamp_max_56), kwargs = {})
#   %div_56 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_215, 6), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_56, [2, 3], True), kwargs = {})
#   %convolution_60 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_4, %arg135_1, %arg136_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_60,), kwargs = {})
triton_poi_fused_convolution_hardswish_mean_relu_17 = async_compile.triton('triton_poi_fused_convolution_hardswish_mean_relu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardswish_mean_relu_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ew/cewue4roki3dxkg2cr6l6iasc7ytrpic6xxh5yntu4szp55x3r4y.py
# Topologically Sorted Source Nodes: [x_166, x_se_12, x_se_13, x_se_14, x_se_15, hardsigmoid_3, x_167], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_3 => add_163, clamp_max_57, clamp_min_57, div_57
#   x_166 => add_162, clamp_max_56, clamp_min_56, div_56, mul_215
#   x_167 => mul_216
#   x_se_12 => mean_4
#   x_se_13 => convolution_60
#   x_se_14 => relu_3
#   x_se_15 => convolution_61
# Graph fragment:
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_161, 3), kwargs = {})
#   %clamp_min_56 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_162, 0), kwargs = {})
#   %clamp_max_56 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_56, 6), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %clamp_max_56), kwargs = {})
#   %div_56 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_215, 6), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_56, [2, 3], True), kwargs = {})
#   %convolution_60 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_4, %arg135_1, %arg136_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_60,), kwargs = {})
#   %convolution_61 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %arg137_1, %arg138_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_163 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_61, 3), kwargs = {})
#   %clamp_min_57 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_163, 0), kwargs = {})
#   %clamp_max_57 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_57, 6), kwargs = {})
#   %div_57 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_57, 6), kwargs = {})
#   %mul_216 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_56, %div_57), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_18 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_18(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 12544)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + (256*x2)), xmask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_sahanp/qc/cqcompd344jdxjsc76redu3htul6zai3hktskhwmzsdnypgfkfvq.py
# Topologically Sorted Source Nodes: [x_170, x_171, x_172, x_173], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_170 => add_166, clamp_max_58, clamp_min_58, div_58, mul_220
#   x_171 => mean_5
#   x_172 => convolution_63
#   x_173 => add_167, clamp_max_59, clamp_min_59, div_59, mul_221
# Graph fragment:
#   %add_166 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_165, 3), kwargs = {})
#   %clamp_min_58 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_166, 0), kwargs = {})
#   %clamp_max_58 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_58, 6), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_165, %clamp_max_58), kwargs = {})
#   %div_58 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_220, 6), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_58, [-1, -2], True), kwargs = {})
#   %convolution_63 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_5, %arg144_1, %arg145_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_63, 3), kwargs = {})
#   %clamp_min_59 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_167, 0), kwargs = {})
#   %clamp_max_59 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_59, 6), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_63, %clamp_max_59), kwargs = {})
#   %div_59 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_221, 6), kwargs = {})
triton_poi_fused_convolution_hardswish_mean_19 = async_compile.triton('triton_poi_fused_convolution_hardswish_mean_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardswish_mean_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (8, ), (1, ))
    assert_size_stride(arg3_1, (8, ), (1, ))
    assert_size_stride(arg4_1, (8, ), (1, ))
    assert_size_stride(arg5_1, (8, ), (1, ))
    assert_size_stride(arg6_1, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg7_1, (8, ), (1, ))
    assert_size_stride(arg8_1, (8, ), (1, ))
    assert_size_stride(arg9_1, (8, ), (1, ))
    assert_size_stride(arg10_1, (8, ), (1, ))
    assert_size_stride(arg11_1, (16, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg12_1, (16, ), (1, ))
    assert_size_stride(arg13_1, (16, ), (1, ))
    assert_size_stride(arg14_1, (16, ), (1, ))
    assert_size_stride(arg15_1, (16, ), (1, ))
    assert_size_stride(arg16_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg17_1, (16, ), (1, ))
    assert_size_stride(arg18_1, (16, ), (1, ))
    assert_size_stride(arg19_1, (16, ), (1, ))
    assert_size_stride(arg20_1, (16, ), (1, ))
    assert_size_stride(arg21_1, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg22_1, (32, ), (1, ))
    assert_size_stride(arg23_1, (32, ), (1, ))
    assert_size_stride(arg24_1, (32, ), (1, ))
    assert_size_stride(arg25_1, (32, ), (1, ))
    assert_size_stride(arg26_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg27_1, (32, ), (1, ))
    assert_size_stride(arg28_1, (32, ), (1, ))
    assert_size_stride(arg29_1, (32, ), (1, ))
    assert_size_stride(arg30_1, (32, ), (1, ))
    assert_size_stride(arg31_1, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg32_1, (32, ), (1, ))
    assert_size_stride(arg33_1, (32, ), (1, ))
    assert_size_stride(arg34_1, (32, ), (1, ))
    assert_size_stride(arg35_1, (32, ), (1, ))
    assert_size_stride(arg36_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg37_1, (32, ), (1, ))
    assert_size_stride(arg38_1, (32, ), (1, ))
    assert_size_stride(arg39_1, (32, ), (1, ))
    assert_size_stride(arg40_1, (32, ), (1, ))
    assert_size_stride(arg41_1, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg47_1, (64, ), (1, ))
    assert_size_stride(arg48_1, (64, ), (1, ))
    assert_size_stride(arg49_1, (64, ), (1, ))
    assert_size_stride(arg50_1, (64, ), (1, ))
    assert_size_stride(arg51_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg52_1, (64, ), (1, ))
    assert_size_stride(arg53_1, (64, ), (1, ))
    assert_size_stride(arg54_1, (64, ), (1, ))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg57_1, (64, ), (1, ))
    assert_size_stride(arg58_1, (64, ), (1, ))
    assert_size_stride(arg59_1, (64, ), (1, ))
    assert_size_stride(arg60_1, (64, ), (1, ))
    assert_size_stride(arg61_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (128, ), (1, ))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg82_1, (128, ), (1, ))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg87_1, (128, ), (1, ))
    assert_size_stride(arg88_1, (128, ), (1, ))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (128, ), (1, ))
    assert_size_stride(arg96_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg102_1, (128, ), (1, ))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, ), (1, ))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg112_1, (128, ), (1, ))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg117_1, (128, ), (1, ))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg122_1, (32, ), (1, ))
    assert_size_stride(arg123_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg124_1, (128, ), (1, ))
    assert_size_stride(arg125_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg126_1, (256, ), (1, ))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (256, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg131_1, (256, ), (1, ))
    assert_size_stride(arg132_1, (256, ), (1, ))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (256, ), (1, ))
    assert_size_stride(arg135_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg136_1, (64, ), (1, ))
    assert_size_stride(arg137_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg138_1, (256, ), (1, ))
    assert_size_stride(arg139_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg140_1, (256, ), (1, ))
    assert_size_stride(arg141_1, (256, ), (1, ))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (1280, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg145_1, (1280, ), (1, ))
    assert_size_stride(arg146_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg147_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((8, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 24, 9, grid=grid(24, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 8, 112, 112), (100352, 1, 896, 8))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((8, 8, 112, 112), (100352, 1, 896, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x_89, x_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, buf4, 802816, grid=grid(802816), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf3
        # Topologically Sorted Source Nodes: [x_90, x_91], Original ATen: [aten.hardswish, aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf5, (8, 8, 112, 112), (100352, 1, 896, 8))
        del arg6_1
        buf6 = buf5; del buf5  # reuse
        buf7 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_92, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, buf7, 802816, grid=grid(802816), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        del buf6
        # Topologically Sorted Source Nodes: [x_93, x_94], Original ATen: [aten.hardswish, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del arg11_1
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided_cuda((8, 16, 112, 112), (200704, 1, 1792, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_95, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3.run(buf9, arg12_1, arg13_1, arg14_1, arg15_1, buf10, 1605632, grid=grid(1605632), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del buf9
        # Topologically Sorted Source Nodes: [x_96, x_97], Original ATen: [aten.hardswish, aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg16_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf11, (8, 16, 56, 56), (50176, 1, 896, 16))
        del arg16_1
        del buf10
        buf12 = buf11; del buf11  # reuse
        buf13 = empty_strided_cuda((8, 16, 56, 56), (50176, 1, 896, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_98, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_4.run(buf12, arg17_1, arg18_1, arg19_1, arg20_1, buf13, 401408, grid=grid(401408), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del buf12
        # Topologically Sorted Source Nodes: [x_99, x_100], Original ATen: [aten.hardswish, aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del arg21_1
        buf15 = buf14; del buf14  # reuse
        buf16 = reinterpret_tensor(buf7, (8, 32, 56, 56), (100352, 1, 1792, 32), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_101, x_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5.run(buf15, arg22_1, arg23_1, arg24_1, arg25_1, buf16, 802816, grid=grid(802816), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del buf15
        # Topologically Sorted Source Nodes: [x_102, x_103], Original ATen: [aten.hardswish, aten.convolution]
        buf17 = extern_kernels.convolution(buf16, arg26_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf17, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del arg26_1
        buf18 = buf17; del buf17  # reuse
        buf19 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_104, x_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5.run(buf18, arg27_1, arg28_1, arg29_1, arg30_1, buf19, 802816, grid=grid(802816), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        del buf18
        # Topologically Sorted Source Nodes: [x_105, x_106], Original ATen: [aten.hardswish, aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del arg31_1
        buf21 = buf20; del buf20  # reuse
        buf22 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_107, x_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5.run(buf21, arg32_1, arg33_1, arg34_1, arg35_1, buf22, 802816, grid=grid(802816), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        del buf21
        # Topologically Sorted Source Nodes: [x_108, x_109], Original ATen: [aten.hardswish, aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg36_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf23, (8, 32, 28, 28), (25088, 1, 896, 32))
        del arg36_1
        del buf22
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided_cuda((8, 32, 28, 28), (25088, 1, 896, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_110, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6.run(buf24, arg37_1, arg38_1, arg39_1, arg40_1, buf25, 200704, grid=grid(200704), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        del buf24
        # Topologically Sorted Source Nodes: [x_111, x_112], Original ATen: [aten.hardswish, aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 64, 28, 28), (50176, 1, 1792, 64))
        del arg41_1
        buf27 = buf26; del buf26  # reuse
        buf28 = reinterpret_tensor(buf13, (8, 64, 28, 28), (50176, 1, 1792, 64), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_113, x_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7.run(buf27, arg42_1, arg43_1, arg44_1, arg45_1, buf28, 401408, grid=grid(401408), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        del buf27
        # Topologically Sorted Source Nodes: [x_114, x_115], Original ATen: [aten.hardswish, aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg46_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf29, (8, 64, 28, 28), (50176, 1, 1792, 64))
        del arg46_1
        buf30 = buf29; del buf29  # reuse
        buf31 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_116, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7.run(buf30, arg47_1, arg48_1, arg49_1, arg50_1, buf31, 401408, grid=grid(401408), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        del buf30
        # Topologically Sorted Source Nodes: [x_117, x_118], Original ATen: [aten.hardswish, aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 64, 28, 28), (50176, 1, 1792, 64))
        del arg51_1
        buf33 = buf32; del buf32  # reuse
        buf34 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_119, x_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7.run(buf33, arg52_1, arg53_1, arg54_1, arg55_1, buf34, 401408, grid=grid(401408), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        del buf33
        # Topologically Sorted Source Nodes: [x_120, x_121], Original ATen: [aten.hardswish, aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg56_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf35, (8, 64, 14, 14), (12544, 1, 896, 64))
        del arg56_1
        del buf34
        buf36 = buf35; del buf35  # reuse
        buf37 = empty_strided_cuda((8, 64, 14, 14), (12544, 1, 896, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_122, x_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8.run(buf36, arg57_1, arg58_1, arg59_1, arg60_1, buf37, 100352, grid=grid(100352), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf36
        # Topologically Sorted Source Nodes: [x_123, x_124], Original ATen: [aten.hardswish, aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg61_1
        buf39 = buf38; del buf38  # reuse
        buf40 = reinterpret_tensor(buf25, (8, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_125, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf39, arg62_1, arg63_1, arg64_1, arg65_1, buf40, 200704, grid=grid(200704), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        del buf39
        # Topologically Sorted Source Nodes: [x_126, x_127], Original ATen: [aten.hardswish, aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg66_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf41, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg66_1
        buf42 = buf41; del buf41  # reuse
        buf43 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_128, x_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf42, arg67_1, arg68_1, arg69_1, arg70_1, buf43, 200704, grid=grid(200704), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        del buf42
        # Topologically Sorted Source Nodes: [x_129, x_130], Original ATen: [aten.hardswish, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg71_1
        buf45 = buf44; del buf44  # reuse
        buf46 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_131, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf45, arg72_1, arg73_1, arg74_1, arg75_1, buf46, 200704, grid=grid(200704), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        del buf45
        # Topologically Sorted Source Nodes: [x_132, x_133], Original ATen: [aten.hardswish, aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg76_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf47, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg76_1
        buf48 = buf47; del buf47  # reuse
        buf49 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_134, x_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf48, arg77_1, arg78_1, arg79_1, arg80_1, buf49, 200704, grid=grid(200704), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        del buf48
        # Topologically Sorted Source Nodes: [x_135, x_136], Original ATen: [aten.hardswish, aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg81_1
        buf51 = buf50; del buf50  # reuse
        buf52 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [x_137, x_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf51, arg82_1, arg83_1, arg84_1, arg85_1, buf52, 200704, grid=grid(200704), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        del buf51
        # Topologically Sorted Source Nodes: [x_138, x_139], Original ATen: [aten.hardswish, aten.convolution]
        buf53 = extern_kernels.convolution(buf52, arg86_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf53, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg86_1
        buf54 = buf53; del buf53  # reuse
        buf55 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_140, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf54, arg87_1, arg88_1, arg89_1, arg90_1, buf55, 200704, grid=grid(200704), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        del buf54
        # Topologically Sorted Source Nodes: [x_141, x_142], Original ATen: [aten.hardswish, aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg91_1
        buf57 = buf56; del buf56  # reuse
        buf58 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_143, x_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf57, arg92_1, arg93_1, arg94_1, arg95_1, buf58, 200704, grid=grid(200704), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf57
        # Topologically Sorted Source Nodes: [x_144, x_145], Original ATen: [aten.hardswish, aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg96_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf59, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg96_1
        buf60 = buf59; del buf59  # reuse
        buf61 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_146, x_147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf60, arg97_1, arg98_1, arg99_1, arg100_1, buf61, 200704, grid=grid(200704), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf60
        # Topologically Sorted Source Nodes: [x_147, x_148], Original ATen: [aten.hardswish, aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg101_1
        buf63 = buf62; del buf62  # reuse
        buf64 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_149, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf63, arg102_1, arg103_1, arg104_1, arg105_1, buf64, 200704, grid=grid(200704), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        del buf63
        # Topologically Sorted Source Nodes: [x_150, x_151], Original ATen: [aten.hardswish, aten.convolution]
        buf65 = extern_kernels.convolution(buf64, arg106_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf65, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg106_1
        buf66 = buf65; del buf65  # reuse
        buf67 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_152, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf66, arg107_1, arg108_1, arg109_1, arg110_1, buf67, 200704, grid=grid(200704), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        del buf66
        # Topologically Sorted Source Nodes: [x_153, x_154], Original ATen: [aten.hardswish, aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg111_1
        buf69 = buf68; del buf68  # reuse
        buf70 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_155, x_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf69, arg112_1, arg113_1, arg114_1, arg115_1, buf70, 200704, grid=grid(200704), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        del buf69
        # Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten.hardswish, aten.convolution]
        buf71 = extern_kernels.convolution(buf70, arg116_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf71, (8, 128, 7, 7), (6272, 1, 896, 128))
        del arg116_1
        del buf70
        buf72 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf72, arg117_1, arg118_1, arg119_1, arg120_1, 50176, grid=grid(50176), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        buf74 = empty_strided_cuda((8, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_159, x_se_8], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_11.run(buf72, buf74, 1024, 49, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_159, x_se_8, x_se_9], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg121_1
        del buf74
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_159, x_se_8, x_se_9, x_se_10], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_12.run(buf76, arg122_1, 256, grid=grid(256), stream=stream0)
        del arg122_1
        # Topologically Sorted Source Nodes: [x_159, x_se_8, x_se_9, x_se_10, x_se_11], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        buf77 = extern_kernels.convolution(buf76, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg123_1
        del buf76
        buf78 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_159, x_se_8, x_se_9, x_se_10, x_se_11, hardsigmoid_2, x_160], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_13.run(buf78, buf77, arg124_1, 50176, grid=grid(50176), stream=stream0)
        del arg124_1
        del buf77
        # Topologically Sorted Source Nodes: [x_159, x_se_8, x_se_9, x_se_10, x_se_11, hardsigmoid_2, x_160, x_161], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf79 = extern_kernels.convolution(buf78, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 256, 7, 7), (12544, 1, 1792, 256))
        del arg125_1
        del buf78
        buf80 = buf79; del buf79  # reuse
        buf81 = reinterpret_tensor(buf37, (8, 256, 7, 7), (12544, 1, 1792, 256), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_162, x_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_14.run(buf80, arg126_1, arg127_1, arg128_1, arg129_1, buf81, 100352, grid=grid(100352), stream=stream0)
        del arg126_1
        del arg127_1
        del arg128_1
        del arg129_1
        del buf80
        # Topologically Sorted Source Nodes: [x_163, x_164], Original ATen: [aten.hardswish, aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg130_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf82, (8, 256, 7, 7), (12544, 1, 1792, 256))
        del arg130_1
        del buf81
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [x_165], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf83, arg131_1, arg132_1, arg133_1, arg134_1, 100352, grid=grid(100352), stream=stream0)
        del arg131_1
        del arg132_1
        del arg133_1
        del arg134_1
        buf85 = empty_strided_cuda((8, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_166, x_se_12], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_16.run(buf83, buf85, 2048, 49, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_166, x_se_12, x_se_13], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg135_1
        del buf85
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_166, x_se_12, x_se_13, x_se_14], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_17.run(buf87, arg136_1, 512, grid=grid(512), stream=stream0)
        del arg136_1
        # Topologically Sorted Source Nodes: [x_166, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu]
        buf88 = extern_kernels.convolution(buf87, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg137_1
        del buf87
        buf89 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_166, x_se_12, x_se_13, x_se_14, x_se_15, hardsigmoid_3, x_167], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_18.run(buf89, buf88, arg138_1, 100352, grid=grid(100352), stream=stream0)
        del arg138_1
        # Topologically Sorted Source Nodes: [x_166, x_se_12, x_se_13, x_se_14, x_se_15, hardsigmoid_3, x_167, x_168], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.relu, aten.hardsigmoid, aten.mul]
        buf90 = extern_kernels.convolution(buf89, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 256, 7, 7), (12544, 1, 1792, 256))
        del arg139_1
        del buf89
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_169], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf91, arg140_1, arg141_1, arg142_1, arg143_1, 100352, grid=grid(100352), stream=stream0)
        del arg140_1
        del arg141_1
        del arg142_1
        del arg143_1
        buf93 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_170, x_171], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_16.run(buf91, buf93, 2048, 49, grid=grid(2048), stream=stream0)
        del buf91
        # Topologically Sorted Source Nodes: [x_170, x_171, x_172], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 1280, 1, 1), (1280, 1, 1, 1))
        del arg144_1
        del buf93
        buf95 = reinterpret_tensor(buf94, (8, 1280, 1, 1), (1280, 1, 10240, 10240), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_170, x_171, x_172, x_173], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_19.run(buf95, arg145_1, 10240, grid=grid(10240), stream=stream0)
        del arg145_1
        buf96 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_175], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg147_1, reinterpret_tensor(buf95, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg146_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf96)
        del arg146_1
        del arg147_1
        del buf95
    return (buf96, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((16, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1280, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('lcnet_050', benchmark_compiled_module)
