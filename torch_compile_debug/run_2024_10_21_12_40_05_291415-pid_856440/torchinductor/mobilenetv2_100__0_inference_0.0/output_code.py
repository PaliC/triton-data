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
# Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_152 => convolution_52
# Graph fragment:
#   %convolution_52 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/fn/cfnqc5t7ruox32h42p5hkjg336a6auwfmlumtz3g3aru5iaouonc.py
# Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_152 => convolution_52
# Graph fragment:
#   %convolution_52 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/kt/cktaysd27nuhb6s6annw27nrczjv7ztraplls54klvk44yhrhdsx.py
# Topologically Sorted Source Nodes: [x_153, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_153 => add_115, mul_157, mul_158, sub_52
#   x_154 => clamp_max_35, clamp_min_35
# Graph fragment:
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_417), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_419), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_157, %unsqueeze_421), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_158, %unsqueeze_423), kwargs = {})
#   %clamp_min_35 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_115, 0.0), kwargs = {})
#   %clamp_max_35 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_35, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/u3/cu3xxa5afujbnaiyq35kigtl3nrohqj654bzcy3eqmqkshn5viid.py
# Topologically Sorted Source Nodes: [x_159], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_159 => add_119, mul_163, mul_164, sub_54
# Graph fragment:
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_54, %unsqueeze_433), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_435), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %unsqueeze_437), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %unsqueeze_439), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/f2/cf2rwlpjfbnrykdeuxp73bmcya222c65k6n5nv7p2uywbbzo5pri.py
# Topologically Sorted Source Nodes: [x_161, x_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_161 => add_121, mul_166, mul_167, sub_55
#   x_162 => clamp_max_37, clamp_min_37
# Graph fragment:
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_55, %unsqueeze_441), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_443), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_166, %unsqueeze_445), kwargs = {})
#   %add_121 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_167, %unsqueeze_447), kwargs = {})
#   %clamp_min_37 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_121, 0.0), kwargs = {})
#   %clamp_max_37 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_37, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4p/c4pnnhnt7bgqrdwbk4azdmwjy7etvwnzipaqhy7fmtadp7zjdmjs.py
# Topologically Sorted Source Nodes: [x_164, x_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_164 => add_123, mul_169, mul_170, sub_56
#   x_165 => clamp_max_38, clamp_min_38
# Graph fragment:
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_56, %unsqueeze_449), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %unsqueeze_451), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_169, %unsqueeze_453), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_170, %unsqueeze_455), kwargs = {})
#   %clamp_min_38 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_123, 0.0), kwargs = {})
#   %clamp_max_38 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_38, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pe/cpeeb57i23ji6kcxdbrcfll4ge3fxlyru4oze74dlfhunlx4m5cs.py
# Topologically Sorted Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_167 => add_125, mul_172, mul_173, sub_57
# Graph fragment:
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_57, %unsqueeze_457), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_459), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %unsqueeze_461), kwargs = {})
#   %add_125 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %unsqueeze_463), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/hj/chjnccqged3ix36jtj7upcgegnsvnsthacgx4shey5c5neluunhf.py
# Topologically Sorted Source Nodes: [x_169, x_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_169 => add_127, mul_175, mul_176, sub_58
#   x_170 => clamp_max_39, clamp_min_39
# Graph fragment:
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_58, %unsqueeze_465), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_467), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %unsqueeze_469), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %unsqueeze_471), kwargs = {})
#   %clamp_min_39 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_127, 0.0), kwargs = {})
#   %clamp_max_39 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_39, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jc/cjclfqm4qeqbngjhojlq7sx4edkljmviuq33xlxcbzgygdzenwx2.py
# Topologically Sorted Source Nodes: [x_175, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_175 => add_131, mul_181, mul_182, sub_60
#   x_176 => add_132
# Graph fragment:
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_60, %unsqueeze_481), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_483), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_181, %unsqueeze_485), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_182, %unsqueeze_487), kwargs = {})
#   %add_132 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_131, %add_125), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/ft/cftbfx5vmdrft55pgnmubyht4zrxrqesqpx6xtngbuucurxt4fpr.py
# Topologically Sorted Source Nodes: [x_181, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_181 => add_136, mul_187, mul_188, sub_62
#   x_182 => clamp_max_42, clamp_min_42
# Graph fragment:
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_62, %unsqueeze_497), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %unsqueeze_499), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_187, %unsqueeze_501), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_188, %unsqueeze_503), kwargs = {})
#   %clamp_min_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_136, 0.0), kwargs = {})
#   %clamp_max_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_42, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/42/c42fnziz3lyxofkeprtt45d65eteqs3ssb7gt2xyxgy5pc73qopk.py
# Topologically Sorted Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_184 => add_138, mul_190, mul_191, sub_63
# Graph fragment:
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_63, %unsqueeze_505), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_190, %unsqueeze_509), kwargs = {})
#   %add_138 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %unsqueeze_511), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5d/c5dxhdpsdkue5fcc7ctontzgdiizl7qzu2oazegzh22y2cjuk5xg.py
# Topologically Sorted Source Nodes: [x_186, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_186 => add_140, mul_193, mul_194, sub_64
#   x_187 => clamp_max_43, clamp_min_43
# Graph fragment:
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_64, %unsqueeze_513), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_515), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_193, %unsqueeze_517), kwargs = {})
#   %add_140 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_194, %unsqueeze_519), kwargs = {})
#   %clamp_min_43 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_140, 0.0), kwargs = {})
#   %clamp_max_43 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_43, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6r/c6rjppat75syxnnpgy46vetx37l4jp4yakeotsuvxohvr3ohkmpe.py
# Topologically Sorted Source Nodes: [x_192, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_192 => add_144, mul_199, mul_200, sub_66
#   x_193 => add_145
# Graph fragment:
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_66, %unsqueeze_529), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %unsqueeze_533), kwargs = {})
#   %add_144 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %unsqueeze_535), kwargs = {})
#   %add_145 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_144, %add_138), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/2d/c2dbxmml7fa5sqnvharxqqdmmp2j2ri5aennd4kbl7a23wpx4e2z.py
# Topologically Sorted Source Nodes: [x_207, x_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_207 => add_156, mul_214, mul_215, sub_71
#   x_208 => clamp_max_48, clamp_min_48
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_569), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_571), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_214, %unsqueeze_573), kwargs = {})
#   %add_156 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_215, %unsqueeze_575), kwargs = {})
#   %clamp_min_48 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_156, 0.0), kwargs = {})
#   %clamp_max_48 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_48, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yr/cyrdax4cu737iwafcriflphbmazf2wqb7nsiabk55plddqibl2e7.py
# Topologically Sorted Source Nodes: [x_210], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_210 => add_158, mul_217, mul_218, sub_72
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_72, %unsqueeze_577), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_217, %unsqueeze_581), kwargs = {})
#   %add_158 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_218, %unsqueeze_583), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dq/cdq7h2rqolsu3d6ymtcamfjdmwq23merolpl5pbb6pu7z7ds6woy.py
# Topologically Sorted Source Nodes: [x_212, x_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_212 => add_160, mul_220, mul_221, sub_73
#   x_213 => clamp_max_49, clamp_min_49
# Graph fragment:
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_73, %unsqueeze_585), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %unsqueeze_587), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_220, %unsqueeze_589), kwargs = {})
#   %add_160 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_221, %unsqueeze_591), kwargs = {})
#   %clamp_min_49 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_160, 0.0), kwargs = {})
#   %clamp_max_49 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_49, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eh/cehxwilvwe7gebxbrimys3rh6lxij7woo47cpxjy35isifcam7qq.py
# Topologically Sorted Source Nodes: [x_218, x_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_218 => add_164, mul_226, mul_227, sub_75
#   x_219 => add_165
# Graph fragment:
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_75, %unsqueeze_601), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_75, %unsqueeze_603), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_226, %unsqueeze_605), kwargs = {})
#   %add_164 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_227, %unsqueeze_607), kwargs = {})
#   %add_165 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_164, %add_158), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/xh/cxhf3eoncn33vqvyysftizkf7472mwqvrxj6efxd5eri4gfu67i3.py
# Topologically Sorted Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_245 => add_185, mul_253, mul_254, sub_84
# Graph fragment:
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_84, %unsqueeze_673), kwargs = {})
#   %mul_253 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_675), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_253, %unsqueeze_677), kwargs = {})
#   %add_185 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_254, %unsqueeze_679), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
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


# kernel path: /tmp/torchinductor_sahanp/m4/cm4467e2yp6wzgcxopdkoutna4cbyidbovt6dqcf5cqsyj3ylpm7.py
# Topologically Sorted Source Nodes: [x_247, x_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_247 => add_187, mul_256, mul_257, sub_85
#   x_248 => clamp_max_57, clamp_min_57
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_85, %unsqueeze_681), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %unsqueeze_685), kwargs = {})
#   %add_187 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %unsqueeze_687), kwargs = {})
#   %clamp_min_57 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_187, 0.0), kwargs = {})
#   %clamp_max_57 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_57, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 576
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/of/cofd2l4z37pi36fiftzoskplheitqdhcp7zdwqffwmmf7c2op2mh.py
# Topologically Sorted Source Nodes: [x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_253 => add_191, mul_262, mul_263, sub_87
#   x_254 => add_192
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_697), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %unsqueeze_701), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %unsqueeze_703), kwargs = {})
#   %add_192 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_191, %add_185), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
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


# kernel path: /tmp/torchinductor_sahanp/lq/clq77uqqljydgjbtkecntiwtwsnwcjtqiqiarto4jpjz6woaibni.py
# Topologically Sorted Source Nodes: [x_268, x_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_268 => add_203, mul_277, mul_278, sub_92
#   x_269 => clamp_max_62, clamp_min_62
# Graph fragment:
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_92, %unsqueeze_737), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_277, %unsqueeze_741), kwargs = {})
#   %add_203 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_278, %unsqueeze_743), kwargs = {})
#   %clamp_min_62 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_203, 0.0), kwargs = {})
#   %clamp_max_62 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_62, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 576
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mj/cmj7j5mxhf6qn4hnweqdnstxgsrbloepufo56l7ldssjelyyfrqz.py
# Topologically Sorted Source Nodes: [x_271], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_271 => add_205, mul_280, mul_281, sub_93
# Graph fragment:
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_93, %unsqueeze_745), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_747), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_280, %unsqueeze_749), kwargs = {})
#   %add_205 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_281, %unsqueeze_751), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/gv/cgvriwvfagmg7uzmnnsdnlglawygn6k2elanxecyeynbqcq4q7qa.py
# Topologically Sorted Source Nodes: [x_273, x_274], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_273 => add_207, mul_283, mul_284, sub_94
#   x_274 => clamp_max_63, clamp_min_63
# Graph fragment:
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_94, %unsqueeze_753), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_283, %unsqueeze_757), kwargs = {})
#   %add_207 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_284, %unsqueeze_759), kwargs = {})
#   %clamp_min_63 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_207, 0.0), kwargs = {})
#   %clamp_max_63 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_63, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4i/c4iw7hikajg4ptr3sxfvjmxxw5e4hkxmbye5ohqke7x5rxcqpzow.py
# Topologically Sorted Source Nodes: [x_279, x_280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_279 => add_211, mul_289, mul_290, sub_96
#   x_280 => add_212
# Graph fragment:
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_96, %unsqueeze_769), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %unsqueeze_771), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_773), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %unsqueeze_775), kwargs = {})
#   %add_212 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_211, %add_205), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/s5/cs5zhd4xzzjyezahnh3qltlhmci4jvcjv526hemkhtvan7u3fvy6.py
# Topologically Sorted Source Nodes: [x_297], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_297 => add_225, mul_307, mul_308, sub_102
# Graph fragment:
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_102, %unsqueeze_817), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %unsqueeze_819), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_307, %unsqueeze_821), kwargs = {})
#   %add_225 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_308, %unsqueeze_823), kwargs = {})
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
    x0 = xindex % 320
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


# kernel path: /tmp/torchinductor_sahanp/cg/ccg3ggyq6pddcmwipgzwntcusvsd4zys4q7dfvmsr7qfqakcgpc7.py
# Topologically Sorted Source Nodes: [x_299, x_300, x_301], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.mean]
# Source node to ATen node mapping:
#   x_299 => add_227, mul_310, mul_311, sub_103
#   x_300 => clamp_max_69, clamp_min_69
#   x_301 => mean_1
# Graph fragment:
#   %sub_103 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_103, %unsqueeze_825), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_103, %unsqueeze_827), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, %unsqueeze_829), kwargs = {})
#   %add_227 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_311, %unsqueeze_831), kwargs = {})
#   %clamp_min_69 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_227, 0.0), kwargs = {})
#   %clamp_max_69 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_69, 6.0), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%clamp_max_69, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (62720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = 49.0
    tmp25 = tmp23 / tmp24
    tl.store(out_ptr1 + (x3), tmp25, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1 = args
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
    assert_size_stride(arg11_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg12_1, (16, ), (1, ))
    assert_size_stride(arg13_1, (16, ), (1, ))
    assert_size_stride(arg14_1, (16, ), (1, ))
    assert_size_stride(arg15_1, (16, ), (1, ))
    assert_size_stride(arg16_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg17_1, (96, ), (1, ))
    assert_size_stride(arg18_1, (96, ), (1, ))
    assert_size_stride(arg19_1, (96, ), (1, ))
    assert_size_stride(arg20_1, (96, ), (1, ))
    assert_size_stride(arg21_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg22_1, (96, ), (1, ))
    assert_size_stride(arg23_1, (96, ), (1, ))
    assert_size_stride(arg24_1, (96, ), (1, ))
    assert_size_stride(arg25_1, (96, ), (1, ))
    assert_size_stride(arg26_1, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg27_1, (24, ), (1, ))
    assert_size_stride(arg28_1, (24, ), (1, ))
    assert_size_stride(arg29_1, (24, ), (1, ))
    assert_size_stride(arg30_1, (24, ), (1, ))
    assert_size_stride(arg31_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg32_1, (144, ), (1, ))
    assert_size_stride(arg33_1, (144, ), (1, ))
    assert_size_stride(arg34_1, (144, ), (1, ))
    assert_size_stride(arg35_1, (144, ), (1, ))
    assert_size_stride(arg36_1, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg37_1, (144, ), (1, ))
    assert_size_stride(arg38_1, (144, ), (1, ))
    assert_size_stride(arg39_1, (144, ), (1, ))
    assert_size_stride(arg40_1, (144, ), (1, ))
    assert_size_stride(arg41_1, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg42_1, (24, ), (1, ))
    assert_size_stride(arg43_1, (24, ), (1, ))
    assert_size_stride(arg44_1, (24, ), (1, ))
    assert_size_stride(arg45_1, (24, ), (1, ))
    assert_size_stride(arg46_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg47_1, (144, ), (1, ))
    assert_size_stride(arg48_1, (144, ), (1, ))
    assert_size_stride(arg49_1, (144, ), (1, ))
    assert_size_stride(arg50_1, (144, ), (1, ))
    assert_size_stride(arg51_1, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg52_1, (144, ), (1, ))
    assert_size_stride(arg53_1, (144, ), (1, ))
    assert_size_stride(arg54_1, (144, ), (1, ))
    assert_size_stride(arg55_1, (144, ), (1, ))
    assert_size_stride(arg56_1, (32, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg57_1, (32, ), (1, ))
    assert_size_stride(arg58_1, (32, ), (1, ))
    assert_size_stride(arg59_1, (32, ), (1, ))
    assert_size_stride(arg60_1, (32, ), (1, ))
    assert_size_stride(arg61_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg62_1, (192, ), (1, ))
    assert_size_stride(arg63_1, (192, ), (1, ))
    assert_size_stride(arg64_1, (192, ), (1, ))
    assert_size_stride(arg65_1, (192, ), (1, ))
    assert_size_stride(arg66_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg67_1, (192, ), (1, ))
    assert_size_stride(arg68_1, (192, ), (1, ))
    assert_size_stride(arg69_1, (192, ), (1, ))
    assert_size_stride(arg70_1, (192, ), (1, ))
    assert_size_stride(arg71_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg72_1, (32, ), (1, ))
    assert_size_stride(arg73_1, (32, ), (1, ))
    assert_size_stride(arg74_1, (32, ), (1, ))
    assert_size_stride(arg75_1, (32, ), (1, ))
    assert_size_stride(arg76_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg77_1, (192, ), (1, ))
    assert_size_stride(arg78_1, (192, ), (1, ))
    assert_size_stride(arg79_1, (192, ), (1, ))
    assert_size_stride(arg80_1, (192, ), (1, ))
    assert_size_stride(arg81_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg82_1, (192, ), (1, ))
    assert_size_stride(arg83_1, (192, ), (1, ))
    assert_size_stride(arg84_1, (192, ), (1, ))
    assert_size_stride(arg85_1, (192, ), (1, ))
    assert_size_stride(arg86_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg87_1, (32, ), (1, ))
    assert_size_stride(arg88_1, (32, ), (1, ))
    assert_size_stride(arg89_1, (32, ), (1, ))
    assert_size_stride(arg90_1, (32, ), (1, ))
    assert_size_stride(arg91_1, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg92_1, (192, ), (1, ))
    assert_size_stride(arg93_1, (192, ), (1, ))
    assert_size_stride(arg94_1, (192, ), (1, ))
    assert_size_stride(arg95_1, (192, ), (1, ))
    assert_size_stride(arg96_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg97_1, (192, ), (1, ))
    assert_size_stride(arg98_1, (192, ), (1, ))
    assert_size_stride(arg99_1, (192, ), (1, ))
    assert_size_stride(arg100_1, (192, ), (1, ))
    assert_size_stride(arg101_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg102_1, (64, ), (1, ))
    assert_size_stride(arg103_1, (64, ), (1, ))
    assert_size_stride(arg104_1, (64, ), (1, ))
    assert_size_stride(arg105_1, (64, ), (1, ))
    assert_size_stride(arg106_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg107_1, (384, ), (1, ))
    assert_size_stride(arg108_1, (384, ), (1, ))
    assert_size_stride(arg109_1, (384, ), (1, ))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (384, ), (1, ))
    assert_size_stride(arg114_1, (384, ), (1, ))
    assert_size_stride(arg115_1, (384, ), (1, ))
    assert_size_stride(arg116_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg117_1, (64, ), (1, ))
    assert_size_stride(arg118_1, (64, ), (1, ))
    assert_size_stride(arg119_1, (64, ), (1, ))
    assert_size_stride(arg120_1, (64, ), (1, ))
    assert_size_stride(arg121_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg122_1, (384, ), (1, ))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (384, ), (1, ))
    assert_size_stride(arg125_1, (384, ), (1, ))
    assert_size_stride(arg126_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg127_1, (384, ), (1, ))
    assert_size_stride(arg128_1, (384, ), (1, ))
    assert_size_stride(arg129_1, (384, ), (1, ))
    assert_size_stride(arg130_1, (384, ), (1, ))
    assert_size_stride(arg131_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg132_1, (64, ), (1, ))
    assert_size_stride(arg133_1, (64, ), (1, ))
    assert_size_stride(arg134_1, (64, ), (1, ))
    assert_size_stride(arg135_1, (64, ), (1, ))
    assert_size_stride(arg136_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg137_1, (384, ), (1, ))
    assert_size_stride(arg138_1, (384, ), (1, ))
    assert_size_stride(arg139_1, (384, ), (1, ))
    assert_size_stride(arg140_1, (384, ), (1, ))
    assert_size_stride(arg141_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg142_1, (384, ), (1, ))
    assert_size_stride(arg143_1, (384, ), (1, ))
    assert_size_stride(arg144_1, (384, ), (1, ))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg147_1, (64, ), (1, ))
    assert_size_stride(arg148_1, (64, ), (1, ))
    assert_size_stride(arg149_1, (64, ), (1, ))
    assert_size_stride(arg150_1, (64, ), (1, ))
    assert_size_stride(arg151_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg152_1, (384, ), (1, ))
    assert_size_stride(arg153_1, (384, ), (1, ))
    assert_size_stride(arg154_1, (384, ), (1, ))
    assert_size_stride(arg155_1, (384, ), (1, ))
    assert_size_stride(arg156_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg157_1, (384, ), (1, ))
    assert_size_stride(arg158_1, (384, ), (1, ))
    assert_size_stride(arg159_1, (384, ), (1, ))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg162_1, (96, ), (1, ))
    assert_size_stride(arg163_1, (96, ), (1, ))
    assert_size_stride(arg164_1, (96, ), (1, ))
    assert_size_stride(arg165_1, (96, ), (1, ))
    assert_size_stride(arg166_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg167_1, (576, ), (1, ))
    assert_size_stride(arg168_1, (576, ), (1, ))
    assert_size_stride(arg169_1, (576, ), (1, ))
    assert_size_stride(arg170_1, (576, ), (1, ))
    assert_size_stride(arg171_1, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg172_1, (576, ), (1, ))
    assert_size_stride(arg173_1, (576, ), (1, ))
    assert_size_stride(arg174_1, (576, ), (1, ))
    assert_size_stride(arg175_1, (576, ), (1, ))
    assert_size_stride(arg176_1, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg177_1, (96, ), (1, ))
    assert_size_stride(arg178_1, (96, ), (1, ))
    assert_size_stride(arg179_1, (96, ), (1, ))
    assert_size_stride(arg180_1, (96, ), (1, ))
    assert_size_stride(arg181_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg182_1, (576, ), (1, ))
    assert_size_stride(arg183_1, (576, ), (1, ))
    assert_size_stride(arg184_1, (576, ), (1, ))
    assert_size_stride(arg185_1, (576, ), (1, ))
    assert_size_stride(arg186_1, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg187_1, (576, ), (1, ))
    assert_size_stride(arg188_1, (576, ), (1, ))
    assert_size_stride(arg189_1, (576, ), (1, ))
    assert_size_stride(arg190_1, (576, ), (1, ))
    assert_size_stride(arg191_1, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg192_1, (96, ), (1, ))
    assert_size_stride(arg193_1, (96, ), (1, ))
    assert_size_stride(arg194_1, (96, ), (1, ))
    assert_size_stride(arg195_1, (96, ), (1, ))
    assert_size_stride(arg196_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg197_1, (576, ), (1, ))
    assert_size_stride(arg198_1, (576, ), (1, ))
    assert_size_stride(arg199_1, (576, ), (1, ))
    assert_size_stride(arg200_1, (576, ), (1, ))
    assert_size_stride(arg201_1, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg202_1, (576, ), (1, ))
    assert_size_stride(arg203_1, (576, ), (1, ))
    assert_size_stride(arg204_1, (576, ), (1, ))
    assert_size_stride(arg205_1, (576, ), (1, ))
    assert_size_stride(arg206_1, (160, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg207_1, (160, ), (1, ))
    assert_size_stride(arg208_1, (160, ), (1, ))
    assert_size_stride(arg209_1, (160, ), (1, ))
    assert_size_stride(arg210_1, (160, ), (1, ))
    assert_size_stride(arg211_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg212_1, (960, ), (1, ))
    assert_size_stride(arg213_1, (960, ), (1, ))
    assert_size_stride(arg214_1, (960, ), (1, ))
    assert_size_stride(arg215_1, (960, ), (1, ))
    assert_size_stride(arg216_1, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg217_1, (960, ), (1, ))
    assert_size_stride(arg218_1, (960, ), (1, ))
    assert_size_stride(arg219_1, (960, ), (1, ))
    assert_size_stride(arg220_1, (960, ), (1, ))
    assert_size_stride(arg221_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg222_1, (160, ), (1, ))
    assert_size_stride(arg223_1, (160, ), (1, ))
    assert_size_stride(arg224_1, (160, ), (1, ))
    assert_size_stride(arg225_1, (160, ), (1, ))
    assert_size_stride(arg226_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg227_1, (960, ), (1, ))
    assert_size_stride(arg228_1, (960, ), (1, ))
    assert_size_stride(arg229_1, (960, ), (1, ))
    assert_size_stride(arg230_1, (960, ), (1, ))
    assert_size_stride(arg231_1, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg232_1, (960, ), (1, ))
    assert_size_stride(arg233_1, (960, ), (1, ))
    assert_size_stride(arg234_1, (960, ), (1, ))
    assert_size_stride(arg235_1, (960, ), (1, ))
    assert_size_stride(arg236_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg237_1, (160, ), (1, ))
    assert_size_stride(arg238_1, (160, ), (1, ))
    assert_size_stride(arg239_1, (160, ), (1, ))
    assert_size_stride(arg240_1, (160, ), (1, ))
    assert_size_stride(arg241_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg242_1, (960, ), (1, ))
    assert_size_stride(arg243_1, (960, ), (1, ))
    assert_size_stride(arg244_1, (960, ), (1, ))
    assert_size_stride(arg245_1, (960, ), (1, ))
    assert_size_stride(arg246_1, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg247_1, (960, ), (1, ))
    assert_size_stride(arg248_1, (960, ), (1, ))
    assert_size_stride(arg249_1, (960, ), (1, ))
    assert_size_stride(arg250_1, (960, ), (1, ))
    assert_size_stride(arg251_1, (320, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg252_1, (320, ), (1, ))
    assert_size_stride(arg253_1, (320, ), (1, ))
    assert_size_stride(arg254_1, (320, ), (1, ))
    assert_size_stride(arg255_1, (320, ), (1, ))
    assert_size_stride(arg256_1, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg257_1, (1280, ), (1, ))
    assert_size_stride(arg258_1, (1280, ), (1, ))
    assert_size_stride(arg259_1, (1280, ), (1, ))
    assert_size_stride(arg260_1, (1280, ), (1, ))
    assert_size_stride(arg261_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg262_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_153, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        # Topologically Sorted Source Nodes: [x_153, x_154, x_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf4, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del arg6_1
        del buf3
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_2.run(buf5, arg7_1, arg8_1, arg9_1, arg10_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [x_156, x_157, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del arg11_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_159], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf7, arg12_1, arg13_1, arg14_1, arg15_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        # Topologically Sorted Source Nodes: [x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 96, 112, 112), (1204224, 1, 10752, 96))
        del arg16_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_161, x_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_4.run(buf9, arg17_1, arg18_1, arg19_1, arg20_1, 9633792, grid=grid(9633792), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        # Topologically Sorted Source Nodes: [x_161, x_162, x_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg21_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf10, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del arg21_1
        del buf9
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_164, x_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_5.run(buf11, arg22_1, arg23_1, arg24_1, arg25_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        # Topologically Sorted Source Nodes: [x_164, x_165, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg26_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf13, arg27_1, arg28_1, arg29_1, arg30_1, 602112, grid=grid(602112), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        # Topologically Sorted Source Nodes: [x_168], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 144, 56, 56), (451584, 1, 8064, 144))
        del arg31_1
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_169, x_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7.run(buf15, arg32_1, arg33_1, arg34_1, arg35_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        # Topologically Sorted Source Nodes: [x_169, x_170, x_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg36_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf16, (8, 144, 56, 56), (451584, 1, 8064, 144))
        del arg36_1
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_172, x_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7.run(buf17, arg37_1, arg38_1, arg39_1, arg40_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        # Topologically Sorted Source Nodes: [x_172, x_173, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg41_1
        del buf17
        buf19 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_175, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf19, buf18, arg42_1, arg43_1, arg44_1, arg45_1, 602112, grid=grid(602112), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        del buf18
        # Topologically Sorted Source Nodes: [x_175, x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 144, 56, 56), (451584, 1, 8064, 144))
        del arg46_1
        del buf19
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_178, x_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_7.run(buf21, arg47_1, arg48_1, arg49_1, arg50_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        # Topologically Sorted Source Nodes: [x_178, x_179, x_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg51_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf22, (8, 144, 28, 28), (112896, 1, 4032, 144))
        del arg51_1
        del buf21
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_181, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf23, arg52_1, arg53_1, arg54_1, arg55_1, 903168, grid=grid(903168), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        # Topologically Sorted Source Nodes: [x_181, x_182, x_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 32, 28, 28), (25088, 1, 896, 32))
        del arg56_1
        del buf23
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf25, arg57_1, arg58_1, arg59_1, arg60_1, 200704, grid=grid(200704), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        # Topologically Sorted Source Nodes: [x_185], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg61_1
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_186, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf27, arg62_1, arg63_1, arg64_1, arg65_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        # Topologically Sorted Source Nodes: [x_186, x_187, x_188], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg66_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf28, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg66_1
        del buf27
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_189, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf29, arg67_1, arg68_1, arg69_1, arg70_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        # Topologically Sorted Source Nodes: [x_189, x_190, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 32, 28, 28), (25088, 1, 896, 32))
        del arg71_1
        del buf29
        buf31 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_192, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_12.run(buf31, buf30, arg72_1, arg73_1, arg74_1, arg75_1, 200704, grid=grid(200704), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        del buf30
        # Topologically Sorted Source Nodes: [x_194], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg76_1
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_195, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf33, arg77_1, arg78_1, arg79_1, arg80_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        # Topologically Sorted Source Nodes: [x_195, x_196, x_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg81_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf34, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg81_1
        del buf33
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_198, x_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf35, arg82_1, arg83_1, arg84_1, arg85_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        # Topologically Sorted Source Nodes: [x_198, x_199, x_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 32, 28, 28), (25088, 1, 896, 32))
        del arg86_1
        del buf35
        buf37 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_201, x_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_12.run(buf37, buf36, arg87_1, arg88_1, arg89_1, arg90_1, 200704, grid=grid(200704), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        del buf36
        # Topologically Sorted Source Nodes: [x_201, x_202, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg91_1
        del buf37
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_204, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_11.run(buf39, arg92_1, arg93_1, arg94_1, arg95_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        # Topologically Sorted Source Nodes: [x_204, x_205, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg96_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf40, (8, 192, 14, 14), (37632, 1, 2688, 192))
        del arg96_1
        del buf39
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_207, x_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf41, arg97_1, arg98_1, arg99_1, arg100_1, 301056, grid=grid(301056), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        # Topologically Sorted Source Nodes: [x_207, x_208, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 64, 14, 14), (12544, 1, 896, 64))
        del arg101_1
        del buf41
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_210], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_14.run(buf43, arg102_1, arg103_1, arg104_1, arg105_1, 100352, grid=grid(100352), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        # Topologically Sorted Source Nodes: [x_211], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg106_1
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [x_212, x_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf45, arg107_1, arg108_1, arg109_1, arg110_1, 602112, grid=grid(602112), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        # Topologically Sorted Source Nodes: [x_212, x_213, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf46 = extern_kernels.convolution(buf45, arg111_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf46, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg111_1
        del buf45
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_215, x_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf47, arg112_1, arg113_1, arg114_1, arg115_1, 602112, grid=grid(602112), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        # Topologically Sorted Source Nodes: [x_215, x_216, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 64, 14, 14), (12544, 1, 896, 64))
        del arg116_1
        del buf47
        buf49 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_218, x_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf49, buf48, arg117_1, arg118_1, arg119_1, arg120_1, 100352, grid=grid(100352), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        del buf48
        # Topologically Sorted Source Nodes: [x_220], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg121_1
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_221, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf51, arg122_1, arg123_1, arg124_1, arg125_1, 602112, grid=grid(602112), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        # Topologically Sorted Source Nodes: [x_221, x_222, x_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg126_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf52, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg126_1
        del buf51
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_224, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf53, arg127_1, arg128_1, arg129_1, arg130_1, 602112, grid=grid(602112), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        # Topologically Sorted Source Nodes: [x_224, x_225, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 64, 14, 14), (12544, 1, 896, 64))
        del arg131_1
        del buf53
        buf55 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [x_227, x_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf55, buf54, arg132_1, arg133_1, arg134_1, arg135_1, 100352, grid=grid(100352), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        del buf54
        # Topologically Sorted Source Nodes: [x_229], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg136_1
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_230, x_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf57, arg137_1, arg138_1, arg139_1, arg140_1, 602112, grid=grid(602112), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        # Topologically Sorted Source Nodes: [x_230, x_231, x_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf58, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg141_1
        del buf57
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_233, x_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf59, arg142_1, arg143_1, arg144_1, arg145_1, 602112, grid=grid(602112), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        # Topologically Sorted Source Nodes: [x_233, x_234, x_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 64, 14, 14), (12544, 1, 896, 64))
        del arg146_1
        del buf59
        buf61 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_236, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf61, buf60, arg147_1, arg148_1, arg149_1, arg150_1, 100352, grid=grid(100352), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        del buf60
        # Topologically Sorted Source Nodes: [x_236, x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg151_1
        del buf61
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_239, x_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf63, arg152_1, arg153_1, arg154_1, arg155_1, 602112, grid=grid(602112), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        # Topologically Sorted Source Nodes: [x_239, x_240, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg156_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf64, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg156_1
        del buf63
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_242, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf65, arg157_1, arg158_1, arg159_1, arg160_1, 602112, grid=grid(602112), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        # Topologically Sorted Source Nodes: [x_242, x_243, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 96, 14, 14), (18816, 1, 1344, 96))
        del arg161_1
        del buf65
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_17.run(buf67, arg162_1, arg163_1, arg164_1, arg165_1, 150528, grid=grid(150528), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        # Topologically Sorted Source Nodes: [x_246], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 576, 14, 14), (112896, 1, 8064, 576))
        del arg166_1
        buf69 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_247, x_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18.run(buf69, arg167_1, arg168_1, arg169_1, arg170_1, 903168, grid=grid(903168), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        # Topologically Sorted Source Nodes: [x_247, x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg171_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf70, (8, 576, 14, 14), (112896, 1, 8064, 576))
        del arg171_1
        del buf69
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_250, x_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18.run(buf71, arg172_1, arg173_1, arg174_1, arg175_1, 903168, grid=grid(903168), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        # Topologically Sorted Source Nodes: [x_250, x_251, x_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 96, 14, 14), (18816, 1, 1344, 96))
        del arg176_1
        del buf71
        buf73 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_19.run(buf73, buf72, arg177_1, arg178_1, arg179_1, arg180_1, 150528, grid=grid(150528), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        del buf72
        # Topologically Sorted Source Nodes: [x_255], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 576, 14, 14), (112896, 1, 8064, 576))
        del arg181_1
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18.run(buf75, arg182_1, arg183_1, arg184_1, arg185_1, 903168, grid=grid(903168), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        # Topologically Sorted Source Nodes: [x_256, x_257, x_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg186_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf76, (8, 576, 14, 14), (112896, 1, 8064, 576))
        del arg186_1
        del buf75
        buf77 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_259, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18.run(buf77, arg187_1, arg188_1, arg189_1, arg190_1, 903168, grid=grid(903168), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        # Topologically Sorted Source Nodes: [x_259, x_260, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 96, 14, 14), (18816, 1, 1344, 96))
        del arg191_1
        del buf77
        buf79 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_262, x_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_19.run(buf79, buf78, arg192_1, arg193_1, arg194_1, arg195_1, 150528, grid=grid(150528), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        del buf78
        # Topologically Sorted Source Nodes: [x_262, x_263, x_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 576, 14, 14), (112896, 1, 8064, 576))
        del arg196_1
        del buf79
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_265, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_18.run(buf81, arg197_1, arg198_1, arg199_1, arg200_1, 903168, grid=grid(903168), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        # Topologically Sorted Source Nodes: [x_265, x_266, x_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg201_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf82, (8, 576, 7, 7), (28224, 1, 4032, 576))
        del arg201_1
        del buf81
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [x_268, x_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_20.run(buf83, arg202_1, arg203_1, arg204_1, arg205_1, 225792, grid=grid(225792), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        # Topologically Sorted Source Nodes: [x_268, x_269, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf84 = extern_kernels.convolution(buf83, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 160, 7, 7), (7840, 1, 1120, 160))
        del arg206_1
        del buf83
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_271], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf85, arg207_1, arg208_1, arg209_1, arg210_1, 62720, grid=grid(62720), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        # Topologically Sorted Source Nodes: [x_272], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del arg211_1
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_273, x_274], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22.run(buf87, arg212_1, arg213_1, arg214_1, arg215_1, 376320, grid=grid(376320), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        # Topologically Sorted Source Nodes: [x_273, x_274, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg216_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf88, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del arg216_1
        del buf87
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_276, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22.run(buf89, arg217_1, arg218_1, arg219_1, arg220_1, 376320, grid=grid(376320), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        # Topologically Sorted Source Nodes: [x_276, x_277, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 160, 7, 7), (7840, 1, 1120, 160))
        del arg221_1
        del buf89
        buf91 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_279, x_280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf91, buf90, arg222_1, arg223_1, arg224_1, arg225_1, 62720, grid=grid(62720), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        del buf90
        # Topologically Sorted Source Nodes: [x_281], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del arg226_1
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_282, x_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22.run(buf93, arg227_1, arg228_1, arg229_1, arg230_1, 376320, grid=grid(376320), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        # Topologically Sorted Source Nodes: [x_282, x_283, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg231_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf94, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del arg231_1
        del buf93
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_285, x_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22.run(buf95, arg232_1, arg233_1, arg234_1, arg235_1, 376320, grid=grid(376320), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        # Topologically Sorted Source Nodes: [x_285, x_286, x_287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 160, 7, 7), (7840, 1, 1120, 160))
        del arg236_1
        del buf95
        buf97 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_288, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf97, buf96, arg237_1, arg238_1, arg239_1, arg240_1, 62720, grid=grid(62720), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        del buf96
        # Topologically Sorted Source Nodes: [x_288, x_289, x_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf98 = extern_kernels.convolution(buf97, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del arg241_1
        del buf97
        buf99 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [x_291, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22.run(buf99, arg242_1, arg243_1, arg244_1, arg245_1, 376320, grid=grid(376320), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        # Topologically Sorted Source Nodes: [x_291, x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg246_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf100, (8, 960, 7, 7), (47040, 1, 6720, 960))
        del arg246_1
        del buf99
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_294, x_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_22.run(buf101, arg247_1, arg248_1, arg249_1, arg250_1, 376320, grid=grid(376320), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        # Topologically Sorted Source Nodes: [x_294, x_295, x_296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 320, 7, 7), (15680, 1, 2240, 320))
        del arg251_1
        del buf101
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_297], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf103, arg252_1, arg253_1, arg254_1, arg255_1, 125440, grid=grid(125440), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        # Topologically Sorted Source Nodes: [x_297, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
        del arg256_1
        del buf103
        buf106 = empty_strided_cuda((8, 1280, 1, 1), (1280, 1, 10240, 10240), torch.float32)
        # Topologically Sorted Source Nodes: [x_299, x_300, x_301], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardtanh_mean_25.run(buf104, arg257_1, arg258_1, arg259_1, arg260_1, buf106, 10240, 49, grid=grid(10240), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        del buf104
        buf107 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_303], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg262_1, reinterpret_tensor(buf106, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg261_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf107)
        del arg261_1
        del arg262_1
        del buf106
    return (buf107, )


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
    arg11_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((32, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((160, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((320, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenetv2_100', benchmark_compiled_module)
