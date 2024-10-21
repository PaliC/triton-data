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
# Topologically Sorted Source Nodes: [x_159], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_159 => convolution_75
# Graph fragment:
#   %convolution_75 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
# Topologically Sorted Source Nodes: [x_159], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_159 => convolution_75
# Graph fragment:
#   %convolution_75 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/vd/cvdmic4lyury7o2ezac5f5ooirpantoqtokalkwch7tz3yqldo3r.py
# Topologically Sorted Source Nodes: [x_160, x_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_160 => add_136, mul_217, mul_218, sub_62
#   x_161 => mul_219, sigmoid_30
# Graph fragment:
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_75, %unsqueeze_497), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %unsqueeze_499), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_217, %unsqueeze_501), kwargs = {})
#   %add_136 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_218, %unsqueeze_503), kwargs = {})
#   %sigmoid_30 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_136,), kwargs = {})
#   %mul_219 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_136, %sigmoid_30), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jd/cjdtli4cufdhodjl5o365jvdchmolynxt22egc75tr5he7i24krd.py
# Topologically Sorted Source Nodes: [x_163, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_163 => add_138, mul_221, mul_222, sub_63
#   x_164 => clamp_max_16, clamp_min_16
# Graph fragment:
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_76, %unsqueeze_505), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_221, %unsqueeze_509), kwargs = {})
#   %add_138 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_222, %unsqueeze_511), kwargs = {})
#   %clamp_min_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_138, 0.0), kwargs = {})
#   %clamp_max_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_16, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/fc/cfcbfwlfwl62z2skqbbequ53vity62cnmuvrnyfgvgoa2tjuinbs.py
# Topologically Sorted Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_166 => add_140, mul_224, mul_225, sub_64
# Graph fragment:
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_513), kwargs = {})
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_515), kwargs = {})
#   %mul_225 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_224, %unsqueeze_517), kwargs = {})
#   %add_140 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_225, %unsqueeze_519), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/zj/czjjq5yzevz6qareqlolwk7exkzilu7wey7rzi4vkwzujec4kmug.py
# Topologically Sorted Source Nodes: [x_168, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_168 => add_142, mul_227, mul_228, sub_65
#   x_169 => mul_229, sigmoid_31
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_78, %unsqueeze_521), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_228 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_227, %unsqueeze_525), kwargs = {})
#   %add_142 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_228, %unsqueeze_527), kwargs = {})
#   %sigmoid_31 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_142,), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_142, %sigmoid_31), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mt/cmtrsife2l5oo32grf2zr2yze5vrfy5nlekima322fdeghuzmse6.py
# Topologically Sorted Source Nodes: [x_171, x_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_171 => add_144, mul_231, mul_232, sub_66
#   x_172 => clamp_max_17, clamp_min_17
# Graph fragment:
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_79, %unsqueeze_529), kwargs = {})
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_231, %unsqueeze_533), kwargs = {})
#   %add_144 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_232, %unsqueeze_535), kwargs = {})
#   %clamp_min_17 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_144, 0.0), kwargs = {})
#   %clamp_max_17 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_17, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/cv/ccv6btcsj4tql6x4hrwta7rknae2rizleyapmwjmxuiz6r6dc2bf.py
# Topologically Sorted Source Nodes: [x_174], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_174 => add_146, mul_234, mul_235, sub_67
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_80, %unsqueeze_537), kwargs = {})
#   %mul_234 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_234, %unsqueeze_541), kwargs = {})
#   %add_146 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_235, %unsqueeze_543), kwargs = {})
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
    xnumel = 677376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 27
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


# kernel path: /tmp/torchinductor_sahanp/64/c64el4yijdbdigmbuwvmflji3q5in6hf2owp5plv4xwvbjo56udd.py
# Topologically Sorted Source Nodes: [x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_176 => add_148, mul_237, mul_238, sub_68
#   x_177 => mul_239, sigmoid_32
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_81, %unsqueeze_545), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_237, %unsqueeze_549), kwargs = {})
#   %add_148 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_238, %unsqueeze_551), kwargs = {})
#   %sigmoid_32 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_148,), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_148, %sigmoid_32), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4064256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 162
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gb/cgbggonf3gsynfzrs5bpkj2j67yp2ygwmmgczlq4kcui2u343q2a.py
# Topologically Sorted Source Nodes: [x_179, x_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   x_179 => add_150, mul_241, mul_242, sub_69
#   x_180 => clamp_max_18, clamp_min_18
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_82, %unsqueeze_553), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_241, %unsqueeze_557), kwargs = {})
#   %add_150 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_242, %unsqueeze_559), kwargs = {})
#   %clamp_min_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_150, 0.0), kwargs = {})
#   %clamp_max_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_18, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4064256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 162
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


# kernel path: /tmp/torchinductor_sahanp/m6/cm6wdahjma2grdva4a3smmtfxl3wn2ctrw5m3indqejsasxc7io5.py
# Topologically Sorted Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_182 => add_152, mul_244, mul_245, sub_70
# Graph fragment:
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_83, %unsqueeze_561), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_563), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_244, %unsqueeze_565), kwargs = {})
#   %add_152 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_245, %unsqueeze_567), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 38
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


# kernel path: /tmp/torchinductor_sahanp/3s/c3szapnith65fgocebrhmoisyvk7vsyltzv57mz53mzqsbp5mldk.py
# Topologically Sorted Source Nodes: [x_183], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_183 => cat_11
# Graph fragment:
#   %cat_11 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_153, %slice_48], 1), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 38
    x1 = (xindex // 38)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 27, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((38*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((27*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 38, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (27 + (38*x1) + ((-27) + x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hi/chir4mrz5qnufhhaqxdroclif4cjw464how2ft4joyc2onytqzeu.py
# Topologically Sorted Source Nodes: [x_185, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_185 => add_155, mul_247, mul_248, sub_71
#   x_186 => mul_249, sigmoid_33
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_84, %unsqueeze_569), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_571), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_247, %unsqueeze_573), kwargs = {})
#   %add_155 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_248, %unsqueeze_575), kwargs = {})
#   %sigmoid_33 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_155,), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_155, %sigmoid_33), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5720064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 228
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/et/cetkcbcbgg2j5sktgne6cvvm2i5jedsyudpdi5bil6txevux5bqf.py
# Topologically Sorted Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_188 => add_157, mul_251, mul_252, sub_72
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_85, %unsqueeze_577), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_251, %unsqueeze_581), kwargs = {})
#   %add_157 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_252, %unsqueeze_583), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1430016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 228
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


# kernel path: /tmp/torchinductor_sahanp/7m/c7mvl66zusly3xp3kdym2yvkaav33xvh5725n6uqxwgxfhfucqbo.py
# Topologically Sorted Source Nodes: [x_se_52], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_52 => mean_14
# Graph fragment:
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_157, [2, 3], True), kwargs = {})
triton_red_fused_mean_14 = async_compile.triton('triton_red_fused_mean_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_14(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12768
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 228
    x1 = (xindex // 228)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (228*r2) + (25536*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lc/clcdtr2c3wvymmxook2ptmrhychmrpl7ziu7osvevd6a34sbqvsu.py
# Topologically Sorted Source Nodes: [x_se_52], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_52 => mean_14
# Graph fragment:
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_157, [2, 3], True), kwargs = {})
triton_per_fused_mean_15 = async_compile.triton('triton_per_fused_mean_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_15(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1824
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 228
    x1 = (xindex // 228)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (228*r2) + (1596*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mc/cmcvroqdy75h6d3755s3mnmmubbypqcmro7e5i54m3wpdocl6dvl.py
# Topologically Sorted Source Nodes: [x_se_52, x_se_53, batch_norm_73, x_se_54], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_73 => add_159, mul_254, mul_255, sub_73
#   x_se_52 => mean_14
#   x_se_53 => convolution_86
#   x_se_54 => relu_13
# Graph fragment:
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_157, [2, 3], True), kwargs = {})
#   %convolution_86 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_14, %arg56_1, %arg57_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_86, %unsqueeze_585), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %unsqueeze_587), kwargs = {})
#   %mul_255 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_254, %unsqueeze_589), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_255, %unsqueeze_591), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_159,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 19
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ys/cyssxszkqrrczrsy2bpo4rnpxoa4qoaizteb2g66cgoy53oqm62q.py
# Topologically Sorted Source Nodes: [x_se_52, x_se_53, batch_norm_73, x_se_54, x_se_55, sigmoid_13, x_189, x_190], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_73 => add_159, mul_254, mul_255, sub_73
#   sigmoid_13 => sigmoid_34
#   x_189 => mul_256
#   x_190 => clamp_max_19, clamp_min_19
#   x_se_52 => mean_14
#   x_se_53 => convolution_86
#   x_se_54 => relu_13
#   x_se_55 => convolution_87
# Graph fragment:
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_157, [2, 3], True), kwargs = {})
#   %convolution_86 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_14, %arg56_1, %arg57_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_86, %unsqueeze_585), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %unsqueeze_587), kwargs = {})
#   %mul_255 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_254, %unsqueeze_589), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_255, %unsqueeze_591), kwargs = {})
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_159,), kwargs = {})
#   %convolution_87 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg62_1, %arg63_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_34 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_87,), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_157, %sigmoid_34), kwargs = {})
#   %clamp_min_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_256, 0.0), kwargs = {})
#   %clamp_max_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_19, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_17(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1430016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 228
    x2 = (xindex // 178752)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (228*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ux/cux7mssihc3akand3js3enxyeza75fewwv3oclfycpwyuvwaftcr.py
# Topologically Sorted Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_192 => add_161, mul_258, mul_259, sub_74
# Graph fragment:
#   %sub_74 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_88, %unsqueeze_593), kwargs = {})
#   %mul_258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_74, %unsqueeze_595), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_258, %unsqueeze_597), kwargs = {})
#   %add_161 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_259, %unsqueeze_599), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 313600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 50
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


# kernel path: /tmp/torchinductor_sahanp/vp/cvpb6bmljujygpzbwvgx5lkpotzskevply44w2sqhtv3jhxnouo5.py
# Topologically Sorted Source Nodes: [x_194, x_195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_194 => add_163, mul_261, mul_262, sub_75
#   x_195 => mul_263, sigmoid_35
# Graph fragment:
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_89, %unsqueeze_601), kwargs = {})
#   %mul_261 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_75, %unsqueeze_603), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_261, %unsqueeze_605), kwargs = {})
#   %add_163 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_262, %unsqueeze_607), kwargs = {})
#   %sigmoid_35 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_163,), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_163, %sigmoid_35), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 300
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/um/cumclve36uoxn32gk2i3dsfjwdcugntzw2cdnc2bexjjqylkb5cl.py
# Topologically Sorted Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_197 => add_165, mul_265, mul_266, sub_76
# Graph fragment:
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_90, %unsqueeze_609), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %unsqueeze_611), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_265, %unsqueeze_613), kwargs = {})
#   %add_165 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_266, %unsqueeze_615), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 300
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


# kernel path: /tmp/torchinductor_sahanp/ln/clnqk2w4lbkgj6wgovh5gftidfsvjxfr45ndyyzvprj2cf5mtdyg.py
# Topologically Sorted Source Nodes: [x_se_56], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_56 => mean_15
# Graph fragment:
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_165, [2, 3], True), kwargs = {})
triton_red_fused_mean_21 = async_compile.triton('triton_red_fused_mean_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_21(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16800
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 300
    x1 = (xindex // 300)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (300*r2) + (33600*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tc/ctcsznfiytj6ubixpcpqavn2tfydooqczcgpgfpe63lzpnrkfbvb.py
# Topologically Sorted Source Nodes: [x_se_56], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_56 => mean_15
# Graph fragment:
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_165, [2, 3], True), kwargs = {})
triton_per_fused_mean_22 = async_compile.triton('triton_per_fused_mean_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_22(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2400
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 300
    x1 = (xindex // 300)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (300*r2) + (2100*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/et/cetpczkqh7m56cdm5n4qfenc4u2tnlhulwxrevvb77evataokdng.py
# Topologically Sorted Source Nodes: [x_se_56, x_se_57, batch_norm_77, x_se_58], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_77 => add_167, mul_268, mul_269, sub_77
#   x_se_56 => mean_15
#   x_se_57 => convolution_91
#   x_se_58 => relu_14
# Graph fragment:
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_165, [2, 3], True), kwargs = {})
#   %convolution_91 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_15, %arg79_1, %arg80_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_91, %unsqueeze_617), kwargs = {})
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_268, %unsqueeze_621), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_269, %unsqueeze_623), kwargs = {})
#   %relu_14 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_167,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rk/crkwqllxhcvijcsvzwf6j3qefreng4acn47zl3j3n2emrrgymtrf.py
# Topologically Sorted Source Nodes: [x_se_56, x_se_57, batch_norm_77, x_se_58, x_se_59, sigmoid_14, x_198, x_199], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_77 => add_167, mul_268, mul_269, sub_77
#   sigmoid_14 => sigmoid_36
#   x_198 => mul_270
#   x_199 => clamp_max_20, clamp_min_20
#   x_se_56 => mean_15
#   x_se_57 => convolution_91
#   x_se_58 => relu_14
#   x_se_59 => convolution_92
# Graph fragment:
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_165, [2, 3], True), kwargs = {})
#   %convolution_91 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_15, %arg79_1, %arg80_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_91, %unsqueeze_617), kwargs = {})
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_268, %unsqueeze_621), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_269, %unsqueeze_623), kwargs = {})
#   %relu_14 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_167,), kwargs = {})
#   %convolution_92 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg85_1, %arg86_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_36 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_92,), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_165, %sigmoid_36), kwargs = {})
#   %clamp_min_20 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_270, 0.0), kwargs = {})
#   %clamp_max_20 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_20, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_24(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 300
    x2 = (xindex // 235200)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (300*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/x6/cx6zpbszbte66b5hv5esf4j3bwzkwjzcbb7ncewbb6fufffumptr.py
# Topologically Sorted Source Nodes: [x_201], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_201 => add_169, mul_272, mul_273, sub_78
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_93, %unsqueeze_625), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_272, %unsqueeze_629), kwargs = {})
#   %add_169 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_273, %unsqueeze_631), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 382592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 61
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


# kernel path: /tmp/torchinductor_sahanp/mx/cmxrldzybfs7srnd3e7j2jmwp2txb77pm6axzpjmkuhssinlacs7.py
# Topologically Sorted Source Nodes: [x_202], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_202 => cat_12
# Graph fragment:
#   %cat_12 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_170, %slice_52], 1), kwargs = {})
triton_poi_fused_cat_26 = async_compile.triton('triton_poi_fused_cat_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_26(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 382592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 61
    x1 = (xindex // 61)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 50, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((61*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((50*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 61, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (50 + (61*x1) + ((-50) + x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cc/ccc3wdb372ioecwj6pwkpkhrtxechrppigsbsungz5jhczerzgmb.py
# Topologically Sorted Source Nodes: [x_204, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_204 => add_172, mul_275, mul_276, sub_79
#   x_205 => mul_277, sigmoid_37
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_94, %unsqueeze_633), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_635), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_275, %unsqueeze_637), kwargs = {})
#   %add_172 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_276, %unsqueeze_639), kwargs = {})
#   %sigmoid_37 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_172,), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_172, %sigmoid_37), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2295552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 366
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ga/cgay245wpt4lsip6eovaywzgff42ogmvwytgubmofkrj7mcryprf.py
# Topologically Sorted Source Nodes: [x_207], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_207 => add_174, mul_279, mul_280, sub_80
# Graph fragment:
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_95, %unsqueeze_641), kwargs = {})
#   %mul_279 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_643), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_279, %unsqueeze_645), kwargs = {})
#   %add_174 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_280, %unsqueeze_647), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 573888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 366
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


# kernel path: /tmp/torchinductor_sahanp/ri/crizffaguenka27jumqao4hhnzdv373mcduc3qvzvejhfnzhi6am.py
# Topologically Sorted Source Nodes: [x_se_60], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_60 => mean_16
# Graph fragment:
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_174, [2, 3], True), kwargs = {})
triton_red_fused_mean_29 = async_compile.triton('triton_red_fused_mean_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_29(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5856
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 366
    x1 = (xindex // 366)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (366*r2) + (35868*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bo/cbozcjgmr2bmh4ebxjxcg6uk6kjpbtdfpubmqrdeyk6flrx54zdp.py
# Topologically Sorted Source Nodes: [x_se_60], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_60 => mean_16
# Graph fragment:
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_174, [2, 3], True), kwargs = {})
triton_per_fused_mean_30 = async_compile.triton('triton_per_fused_mean_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_30(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2928
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 366
    x1 = (xindex // 366)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (366*r2) + (732*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/t3/ct3257gfzkgpg72w27k4b5bw7vcifcfbiqd2tcbi5ig6cmwmb6xp.py
# Topologically Sorted Source Nodes: [x_se_60, x_se_61, batch_norm_81, x_se_62], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_81 => add_176, mul_282, mul_283, sub_81
#   x_se_60 => mean_16
#   x_se_61 => convolution_96
#   x_se_62 => relu_15
# Graph fragment:
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_174, [2, 3], True), kwargs = {})
#   %convolution_96 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_16, %arg102_1, %arg103_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_96, %unsqueeze_649), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_651), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_282, %unsqueeze_653), kwargs = {})
#   %add_176 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_283, %unsqueeze_655), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_176,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 30
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rc/crcidabmelhrb4dr4cht2kn7q5oo7bdbno5jakdn6oxjvv5sbgj2.py
# Topologically Sorted Source Nodes: [x_se_60, x_se_61, batch_norm_81, x_se_62, x_se_63, sigmoid_15, x_208, x_209], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_81 => add_176, mul_282, mul_283, sub_81
#   sigmoid_15 => sigmoid_38
#   x_208 => mul_284
#   x_209 => clamp_max_21, clamp_min_21
#   x_se_60 => mean_16
#   x_se_61 => convolution_96
#   x_se_62 => relu_15
#   x_se_63 => convolution_97
# Graph fragment:
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_174, [2, 3], True), kwargs = {})
#   %convolution_96 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_16, %arg102_1, %arg103_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_96, %unsqueeze_649), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_651), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_282, %unsqueeze_653), kwargs = {})
#   %add_176 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_283, %unsqueeze_655), kwargs = {})
#   %relu_15 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_176,), kwargs = {})
#   %convolution_97 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %arg108_1, %arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_38 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_97,), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_174, %sigmoid_38), kwargs = {})
#   %clamp_min_21 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_284, 0.0), kwargs = {})
#   %clamp_max_21 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_21, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_32(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 573888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 366
    x2 = (xindex // 71736)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (366*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rd/crdxj4vagpiweo5wwm6dbxyktadzeoht6pv5y7fghdcmne7d4v7r.py
# Topologically Sorted Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_211 => add_178, mul_286, mul_287, sub_82
# Graph fragment:
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_98, %unsqueeze_657), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_82, %unsqueeze_659), kwargs = {})
#   %mul_287 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_286, %unsqueeze_661), kwargs = {})
#   %add_178 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_287, %unsqueeze_663), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uc/cuciollxfbl2fgk246z2mve6afbpcrlfmlzcwxefp7dqotfqenjc.py
# Topologically Sorted Source Nodes: [x_213, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_213 => add_180, mul_289, mul_290, sub_83
#   x_214 => mul_291, sigmoid_39
# Graph fragment:
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_99, %unsqueeze_665), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_667), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_669), kwargs = {})
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %unsqueeze_671), kwargs = {})
#   %sigmoid_39 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_180,), kwargs = {})
#   %mul_291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_180, %sigmoid_39), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 677376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 432
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ol/coltcz6tecfvqv7ljr4v3orgadpgtbp7ex535jqz2s32rvfmrb45.py
# Topologically Sorted Source Nodes: [x_216], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_216 => add_182, mul_293, mul_294, sub_84
# Graph fragment:
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_673), kwargs = {})
#   %mul_293 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_675), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_293, %unsqueeze_677), kwargs = {})
#   %add_182 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_294, %unsqueeze_679), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 677376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 432
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


# kernel path: /tmp/torchinductor_sahanp/ey/cey2dryewhsq6lpdtgzqvsluhcgvv6whc2p2qtramynmf225cx4h.py
# Topologically Sorted Source Nodes: [x_se_64], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_64 => mean_17
# Graph fragment:
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_182, [2, 3], True), kwargs = {})
triton_red_fused_mean_36 = async_compile.triton('triton_red_fused_mean_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_36(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6912
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 432
    x1 = (xindex // 432)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (432*r2) + (42336*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fx/cfxwodu4mopvbv5w6vb5a5tgrx3f7gu6m4h2s7dgu3wpc6rlz7m5.py
# Topologically Sorted Source Nodes: [x_se_64], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_64 => mean_17
# Graph fragment:
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_182, [2, 3], True), kwargs = {})
triton_per_fused_mean_37 = async_compile.triton('triton_per_fused_mean_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_37(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3456
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 432
    x1 = (xindex // 432)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (432*r2) + (864*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d4/cd4pi5r63zokgtqiqergxwk5vgnvgoac43wisa4bzbjwu4vhlggt.py
# Topologically Sorted Source Nodes: [x_se_64, x_se_65, batch_norm_85, x_se_66], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_85 => add_184, mul_296, mul_297, sub_85
#   x_se_64 => mean_17
#   x_se_65 => convolution_101
#   x_se_66 => relu_16
# Graph fragment:
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_182, [2, 3], True), kwargs = {})
#   %convolution_101 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_17, %arg125_1, %arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_101, %unsqueeze_681), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_297 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_296, %unsqueeze_685), kwargs = {})
#   %add_184 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_297, %unsqueeze_687), kwargs = {})
#   %relu_16 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_184,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 36
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/by/cbyt4sysngt64utxryqytslqrxdicjnz7epkim5j4th3y564hevo.py
# Topologically Sorted Source Nodes: [x_se_64, x_se_65, batch_norm_85, x_se_66, x_se_67, sigmoid_16, x_217, x_218], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_85 => add_184, mul_296, mul_297, sub_85
#   sigmoid_16 => sigmoid_40
#   x_217 => mul_298
#   x_218 => clamp_max_22, clamp_min_22
#   x_se_64 => mean_17
#   x_se_65 => convolution_101
#   x_se_66 => relu_16
#   x_se_67 => convolution_102
# Graph fragment:
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_182, [2, 3], True), kwargs = {})
#   %convolution_101 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_17, %arg125_1, %arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_101, %unsqueeze_681), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_297 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_296, %unsqueeze_685), kwargs = {})
#   %add_184 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_297, %unsqueeze_687), kwargs = {})
#   %relu_16 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_184,), kwargs = {})
#   %convolution_102 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_16, %arg131_1, %arg132_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_40 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_102,), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_182, %sigmoid_40), kwargs = {})
#   %clamp_min_22 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_298, 0.0), kwargs = {})
#   %clamp_max_22 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_22, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_39(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 677376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 432
    x2 = (xindex // 84672)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (432*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7d/c7deg3246o3svcgsa4txd4dizeiams4daqdngqtpqqfmfvovit4r.py
# Topologically Sorted Source Nodes: [x_220], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_220 => add_186, mul_300, mul_301, sub_86
# Graph fragment:
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_103, %unsqueeze_689), kwargs = {})
#   %mul_300 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_86, %unsqueeze_691), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_300, %unsqueeze_693), kwargs = {})
#   %add_186 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_301, %unsqueeze_695), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 84
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


# kernel path: /tmp/torchinductor_sahanp/gu/cguix5nl3pun7sjkfbhwxmrhze23rdxmfvy7jjqkqhkkjank7pep.py
# Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_221 => cat_13
# Graph fragment:
#   %cat_13 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_187, %slice_56], 1), kwargs = {})
triton_poi_fused_cat_41 = async_compile.triton('triton_poi_fused_cat_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_41(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 84
    x1 = (xindex // 84)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 72, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((84*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((72*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 84, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (72 + (84*x1) + ((-72) + x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oh/cohszcq2xr5mvqt7cq6mc6q2dff26ymjvgzthdvufhqnhci5dvtm.py
# Topologically Sorted Source Nodes: [x_223, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_223 => add_189, mul_303, mul_304, sub_87
#   x_224 => mul_305, sigmoid_41
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_104, %unsqueeze_697), kwargs = {})
#   %mul_303 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_304 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_303, %unsqueeze_701), kwargs = {})
#   %add_189 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_304, %unsqueeze_703), kwargs = {})
#   %sigmoid_41 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_189,), kwargs = {})
#   %mul_305 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_189, %sigmoid_41), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_42', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 790272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 504
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7n/c7ndmi2e5klyvgqcgegxbvensddwey3d25fl6izto7alpmcttptn.py
# Topologically Sorted Source Nodes: [x_226], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_226 => add_191, mul_307, mul_308, sub_88
# Graph fragment:
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_705), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_707), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_307, %unsqueeze_709), kwargs = {})
#   %add_191 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_308, %unsqueeze_711), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 790272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 504
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


# kernel path: /tmp/torchinductor_sahanp/wp/cwphws6xm6alqyxbve4ysekmb353iraheftvuuguma2yk6nwjyej.py
# Topologically Sorted Source Nodes: [x_se_68], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_68 => mean_18
# Graph fragment:
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_191, [2, 3], True), kwargs = {})
triton_red_fused_mean_44 = async_compile.triton('triton_red_fused_mean_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_44(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8064
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 504
    x1 = (xindex // 504)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (504*r2) + (49392*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kf/ckfedg2t5w6j2qscntxbyuskqn7463ftxa6e3rsdgnxhp2usgkht.py
# Topologically Sorted Source Nodes: [x_se_68], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_68 => mean_18
# Graph fragment:
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_191, [2, 3], True), kwargs = {})
triton_per_fused_mean_45 = async_compile.triton('triton_per_fused_mean_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_45(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4032
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 504
    x1 = (xindex // 504)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (504*r2) + (1008*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cz/cczdbaifv2rfo6b4c2ou3ghiv7w2rmqrzyhgzpiod6b46t5w7cz7.py
# Topologically Sorted Source Nodes: [x_se_68, x_se_69, batch_norm_89, x_se_70], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_89 => add_193, mul_310, mul_311, sub_89
#   x_se_68 => mean_18
#   x_se_69 => convolution_106
#   x_se_70 => relu_17
# Graph fragment:
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_191, [2, 3], True), kwargs = {})
#   %convolution_106 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_18, %arg148_1, %arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_106, %unsqueeze_713), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %unsqueeze_715), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, %unsqueeze_717), kwargs = {})
#   %add_193 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_311, %unsqueeze_719), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_193,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_46', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 42
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rs/crsshnknhppu2l33liaxtujjcsb2jm7fd273ltyxcioyw6b6aefe.py
# Topologically Sorted Source Nodes: [x_se_68, x_se_69, batch_norm_89, x_se_70, x_se_71, sigmoid_17, x_227, x_228], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_89 => add_193, mul_310, mul_311, sub_89
#   sigmoid_17 => sigmoid_42
#   x_227 => mul_312
#   x_228 => clamp_max_23, clamp_min_23
#   x_se_68 => mean_18
#   x_se_69 => convolution_106
#   x_se_70 => relu_17
#   x_se_71 => convolution_107
# Graph fragment:
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_191, [2, 3], True), kwargs = {})
#   %convolution_106 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_18, %arg148_1, %arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_106, %unsqueeze_713), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %unsqueeze_715), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, %unsqueeze_717), kwargs = {})
#   %add_193 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_311, %unsqueeze_719), kwargs = {})
#   %relu_17 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_193,), kwargs = {})
#   %convolution_107 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %arg154_1, %arg155_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_42 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_107,), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_191, %sigmoid_42), kwargs = {})
#   %clamp_min_23 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_312, 0.0), kwargs = {})
#   %clamp_max_23 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_23, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_47(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 790272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 504
    x2 = (xindex // 98784)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (504*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cz/ccz62y3xteboyfwfk4b5dkmbqlzr7fa5r5nk7jo7ur46abr4e3xy.py
# Topologically Sorted Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_230 => add_195, mul_314, mul_315, sub_90
# Graph fragment:
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_108, %unsqueeze_721), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_723), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_314, %unsqueeze_725), kwargs = {})
#   %add_195 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_315, %unsqueeze_727), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 148960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 95
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


# kernel path: /tmp/torchinductor_sahanp/ca/ccajh5ys6hsfc6sazx3cs6mw4rrmuqfyyugcxvxkd7m6faxfjngd.py
# Topologically Sorted Source Nodes: [x_231], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_231 => cat_14
# Graph fragment:
#   %cat_14 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_196, %slice_60], 1), kwargs = {})
triton_poi_fused_cat_49 = async_compile.triton('triton_poi_fused_cat_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_49(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 148960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 95
    x1 = (xindex // 95)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 84, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((95*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((84*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 95, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (84 + (95*x1) + ((-84) + x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/c2/cc2ebkwidu5uzioyolymer3kstuqwvjhjecilrzwo63ey2ew6ku2.py
# Topologically Sorted Source Nodes: [x_233, x_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_233 => add_198, mul_317, mul_318, sub_91
#   x_234 => mul_319, sigmoid_43
# Graph fragment:
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_109, %unsqueeze_729), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %unsqueeze_731), kwargs = {})
#   %mul_318 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_317, %unsqueeze_733), kwargs = {})
#   %add_198 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_318, %unsqueeze_735), kwargs = {})
#   %sigmoid_43 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_198,), kwargs = {})
#   %mul_319 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_198, %sigmoid_43), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 893760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 570
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eu/ceuo322xrnqrl54iy5phngjauzr4hymxswjusdxgzdulmjrsexty.py
# Topologically Sorted Source Nodes: [x_236], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_236 => add_200, mul_321, mul_322, sub_92
# Graph fragment:
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_110, %unsqueeze_737), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_321, %unsqueeze_741), kwargs = {})
#   %add_200 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_322, %unsqueeze_743), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 893760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 570
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


# kernel path: /tmp/torchinductor_sahanp/ts/cts36qz3rtqddejwq2m7rwbufkx2z4b3xcls5crsbfpoqrqbqcpk.py
# Topologically Sorted Source Nodes: [x_se_72], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_72 => mean_19
# Graph fragment:
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_200, [2, 3], True), kwargs = {})
triton_red_fused_mean_52 = async_compile.triton('triton_red_fused_mean_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_52(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9120
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 570
    x1 = (xindex // 570)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (570*r2) + (55860*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g5/cg5boubyq7qatyul5mhvddl4ufl2fwz2w3s7f66yufkfyjh5fvwj.py
# Topologically Sorted Source Nodes: [x_se_72], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_72 => mean_19
# Graph fragment:
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_200, [2, 3], True), kwargs = {})
triton_per_fused_mean_53 = async_compile.triton('triton_per_fused_mean_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_53(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4560
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 570
    x1 = (xindex // 570)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (570*r2) + (1140*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/o6/co6zj3nj23xmvrpgjoxo4lhqj6oxkgwljhxtapp34dnj6goqsvpp.py
# Topologically Sorted Source Nodes: [x_se_72, x_se_73, batch_norm_93, x_se_74], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_93 => add_202, mul_324, mul_325, sub_93
#   x_se_72 => mean_19
#   x_se_73 => convolution_111
#   x_se_74 => relu_18
# Graph fragment:
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_200, [2, 3], True), kwargs = {})
#   %convolution_111 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_19, %arg171_1, %arg172_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_111, %unsqueeze_745), kwargs = {})
#   %mul_324 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_747), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_324, %unsqueeze_749), kwargs = {})
#   %add_202 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_325, %unsqueeze_751), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_202,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_54 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_54', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 47
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/p2/cp26fmxqab5sx6q6f7srroy7idxoe7qbc2wxs3cgomf2eeyn2dut.py
# Topologically Sorted Source Nodes: [x_se_72, x_se_73, batch_norm_93, x_se_74, x_se_75, sigmoid_18, x_237, x_238], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_93 => add_202, mul_324, mul_325, sub_93
#   sigmoid_18 => sigmoid_44
#   x_237 => mul_326
#   x_238 => clamp_max_24, clamp_min_24
#   x_se_72 => mean_19
#   x_se_73 => convolution_111
#   x_se_74 => relu_18
#   x_se_75 => convolution_112
# Graph fragment:
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_200, [2, 3], True), kwargs = {})
#   %convolution_111 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_19, %arg171_1, %arg172_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_111, %unsqueeze_745), kwargs = {})
#   %mul_324 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_747), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_324, %unsqueeze_749), kwargs = {})
#   %add_202 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_325, %unsqueeze_751), kwargs = {})
#   %relu_18 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_202,), kwargs = {})
#   %convolution_112 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %arg177_1, %arg178_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_44 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_112,), kwargs = {})
#   %mul_326 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_200, %sigmoid_44), kwargs = {})
#   %clamp_min_24 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_326, 0.0), kwargs = {})
#   %clamp_max_24 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_24, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_55(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 893760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 570
    x2 = (xindex // 111720)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (570*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pj/cpjodyahxf2ga2g6tjwdnukrmtt7n4lrkobgwa35nwc6wzdphpjl.py
# Topologically Sorted Source Nodes: [x_240], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_240 => add_204, mul_328, mul_329, sub_94
# Graph fragment:
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_113, %unsqueeze_753), kwargs = {})
#   %mul_328 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %mul_329 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_328, %unsqueeze_757), kwargs = {})
#   %add_204 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_329, %unsqueeze_759), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_56', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_56(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 166208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 106
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


# kernel path: /tmp/torchinductor_sahanp/ui/cuidwfsdwqssomc2pg6fbirybwbltt73rgq2qmyuxfhwleukozi2.py
# Topologically Sorted Source Nodes: [x_241], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_241 => cat_15
# Graph fragment:
#   %cat_15 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_205, %slice_64], 1), kwargs = {})
triton_poi_fused_cat_57 = async_compile.triton('triton_poi_fused_cat_57', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_57(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 166208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 106
    x1 = (xindex // 106)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 95, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((106*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((95*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 106, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (95 + (106*x1) + ((-95) + x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fj/cfjdrfuic4al6digde4ufkeunua3e6vtijo2nmwjc7prd5t6o2mt.py
# Topologically Sorted Source Nodes: [x_243, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_243 => add_207, mul_331, mul_332, sub_95
#   x_244 => mul_333, sigmoid_45
# Graph fragment:
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_114, %unsqueeze_761), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %unsqueeze_763), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_331, %unsqueeze_765), kwargs = {})
#   %add_207 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_332, %unsqueeze_767), kwargs = {})
#   %sigmoid_45 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_207,), kwargs = {})
#   %mul_333 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_207, %sigmoid_45), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_58 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_58', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 997248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 636
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bj/cbjy4wk6h5vehlerkcqdiiorsdwnhcuvzbp3igutw5tknqfgvnwy.py
# Topologically Sorted Source Nodes: [x_246], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_246 => add_209, mul_335, mul_336, sub_96
# Graph fragment:
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_769), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %unsqueeze_771), kwargs = {})
#   %mul_336 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_335, %unsqueeze_773), kwargs = {})
#   %add_209 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_336, %unsqueeze_775), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_59 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_59', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 997248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 636
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


# kernel path: /tmp/torchinductor_sahanp/bq/cbq3mdtfhr2jvum5bc5tvwjffpmqbqtnn2ngnws4l3dq5n3gaq3l.py
# Topologically Sorted Source Nodes: [x_se_76], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_76 => mean_20
# Graph fragment:
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_209, [2, 3], True), kwargs = {})
triton_red_fused_mean_60 = async_compile.triton('triton_red_fused_mean_60', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_60(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10176
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 636
    x1 = (xindex // 636)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (636*r2) + (62328*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ng/cngknamm5jvkkjpzpulcwkq3qlf5rnovera2vxabe33gdrq37hx3.py
# Topologically Sorted Source Nodes: [x_se_76], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_76 => mean_20
# Graph fragment:
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_209, [2, 3], True), kwargs = {})
triton_per_fused_mean_61 = async_compile.triton('triton_per_fused_mean_61', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_61(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5088
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 636
    x1 = (xindex // 636)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (636*r2) + (1272*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ky/cky2dojab4jsrfhcsbk5ofy43f5kqgxisdcd2ro4h4gpy62kkivb.py
# Topologically Sorted Source Nodes: [x_se_76, x_se_77, batch_norm_97, x_se_78], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_97 => add_211, mul_338, mul_339, sub_97
#   x_se_76 => mean_20
#   x_se_77 => convolution_116
#   x_se_78 => relu_19
# Graph fragment:
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_209, [2, 3], True), kwargs = {})
#   %convolution_116 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_20, %arg194_1, %arg195_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_97 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_116, %unsqueeze_777), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_97, %unsqueeze_779), kwargs = {})
#   %mul_339 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_338, %unsqueeze_781), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_339, %unsqueeze_783), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_211,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_62 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_62', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_62(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 53
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6w/c6wfhfphydbfhywh4ezsisic7spyjsh7wer5btuunhqe6myu6r2z.py
# Topologically Sorted Source Nodes: [x_se_76, x_se_77, batch_norm_97, x_se_78, x_se_79, sigmoid_19, x_247, x_248], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_97 => add_211, mul_338, mul_339, sub_97
#   sigmoid_19 => sigmoid_46
#   x_247 => mul_340
#   x_248 => clamp_max_25, clamp_min_25
#   x_se_76 => mean_20
#   x_se_77 => convolution_116
#   x_se_78 => relu_19
#   x_se_79 => convolution_117
# Graph fragment:
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_209, [2, 3], True), kwargs = {})
#   %convolution_116 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_20, %arg194_1, %arg195_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_97 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_116, %unsqueeze_777), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_97, %unsqueeze_779), kwargs = {})
#   %mul_339 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_338, %unsqueeze_781), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_339, %unsqueeze_783), kwargs = {})
#   %relu_19 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_211,), kwargs = {})
#   %convolution_117 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_19, %arg200_1, %arg201_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_46 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_117,), kwargs = {})
#   %mul_340 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_209, %sigmoid_46), kwargs = {})
#   %clamp_min_25 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_340, 0.0), kwargs = {})
#   %clamp_max_25 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_25, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_63 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_63', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_63', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_63(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 997248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 636
    x2 = (xindex // 124656)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (636*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4r/c4ri5rieohawk7o7gkdbcbi3ikxw5f5no27hv65zkcwgcgb7kteg.py
# Topologically Sorted Source Nodes: [x_250], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_250 => add_213, mul_342, mul_343, sub_98
# Graph fragment:
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_118, %unsqueeze_785), kwargs = {})
#   %mul_342 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_98, %unsqueeze_787), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_342, %unsqueeze_789), kwargs = {})
#   %add_213 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_343, %unsqueeze_791), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_64 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_64', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_64', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_64(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 183456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 117
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


# kernel path: /tmp/torchinductor_sahanp/i7/ci7pcetgpg3b7je5vanupewmlporshagosm32nocrahc5qvatw5q.py
# Topologically Sorted Source Nodes: [x_251], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_251 => cat_16
# Graph fragment:
#   %cat_16 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_214, %slice_68], 1), kwargs = {})
triton_poi_fused_cat_65 = async_compile.triton('triton_poi_fused_cat_65', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_65(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 183456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 117
    x1 = (xindex // 117)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 106, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((117*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((106*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 117, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (106 + (117*x1) + ((-106) + x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/w5/cw5s46uzswksmj7fglibmyvx3zrefjuazdkifeos43rwcvrrxbnd.py
# Topologically Sorted Source Nodes: [x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_253 => add_216, mul_345, mul_346, sub_99
#   x_254 => mul_347, sigmoid_47
# Graph fragment:
#   %sub_99 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_119, %unsqueeze_793), kwargs = {})
#   %mul_345 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_99, %unsqueeze_795), kwargs = {})
#   %mul_346 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_345, %unsqueeze_797), kwargs = {})
#   %add_216 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_346, %unsqueeze_799), kwargs = {})
#   %sigmoid_47 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_216,), kwargs = {})
#   %mul_347 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_216, %sigmoid_47), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_66 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_66', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_66', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_66(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1100736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 702
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/l3/cl35fzbob6eygygveanr6gfq6ddvrqnkc2qkzd6fslogsgrpvycp.py
# Topologically Sorted Source Nodes: [x_256], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_256 => add_218, mul_349, mul_350, sub_100
# Graph fragment:
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_120, %unsqueeze_801), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_803), kwargs = {})
#   %mul_350 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_349, %unsqueeze_805), kwargs = {})
#   %add_218 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_350, %unsqueeze_807), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_67 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_67', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_67', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_67(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1100736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 702
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


# kernel path: /tmp/torchinductor_sahanp/bu/cbu5swpswiak36fas4sbfpij3hytntdxqm4xr2hml6pcrfcz2oub.py
# Topologically Sorted Source Nodes: [x_se_80], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_80 => mean_21
# Graph fragment:
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_218, [2, 3], True), kwargs = {})
triton_red_fused_mean_68 = async_compile.triton('triton_red_fused_mean_68', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_68(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11232
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 702
    x1 = (xindex // 702)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (702*r2) + (68796*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mm/cmm56s2wj7vjsxjzfrlw4xw7frktfdsrca73ioiky276ikbmc3gy.py
# Topologically Sorted Source Nodes: [x_se_80], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_80 => mean_21
# Graph fragment:
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_218, [2, 3], True), kwargs = {})
triton_per_fused_mean_69 = async_compile.triton('triton_per_fused_mean_69', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_69(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5616
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 702
    x1 = (xindex // 702)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (702*r2) + (1404*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nk/cnkzxnbzaq3vjcxgwichelcsl475bydll2nlwdhgpdh6md2ppyrb.py
# Topologically Sorted Source Nodes: [x_se_80, x_se_81, batch_norm_101, x_se_82], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_101 => add_220, mul_352, mul_353, sub_101
#   x_se_80 => mean_21
#   x_se_81 => convolution_121
#   x_se_82 => relu_20
# Graph fragment:
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_218, [2, 3], True), kwargs = {})
#   %convolution_121 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_21, %arg217_1, %arg218_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_101 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_121, %unsqueeze_809), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_101, %unsqueeze_811), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, %unsqueeze_813), kwargs = {})
#   %add_220 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_353, %unsqueeze_815), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_220,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_70 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_70', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_70', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_70(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 58
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rz/crzhctamxkqd4mzdhlbv73x247bizfgeolvpg6vauiqcon2hq7hv.py
# Topologically Sorted Source Nodes: [x_se_80, x_se_81, batch_norm_101, x_se_82, x_se_83, sigmoid_20, x_257, x_258], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_101 => add_220, mul_352, mul_353, sub_101
#   sigmoid_20 => sigmoid_48
#   x_257 => mul_354
#   x_258 => clamp_max_26, clamp_min_26
#   x_se_80 => mean_21
#   x_se_81 => convolution_121
#   x_se_82 => relu_20
#   x_se_83 => convolution_122
# Graph fragment:
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_218, [2, 3], True), kwargs = {})
#   %convolution_121 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_21, %arg217_1, %arg218_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_101 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_121, %unsqueeze_809), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_101, %unsqueeze_811), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, %unsqueeze_813), kwargs = {})
#   %add_220 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_353, %unsqueeze_815), kwargs = {})
#   %relu_20 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_220,), kwargs = {})
#   %convolution_122 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_20, %arg223_1, %arg224_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_48 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_122,), kwargs = {})
#   %mul_354 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_218, %sigmoid_48), kwargs = {})
#   %clamp_min_26 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_354, 0.0), kwargs = {})
#   %clamp_max_26 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_26, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_71 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_71', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_71', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_71(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1100736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 702
    x2 = (xindex // 137592)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (702*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/iz/ciz3dmz6qm2eqpsdfb5z7p5z4gdrpwwvoop52kvmp5rfmnigv6na.py
# Topologically Sorted Source Nodes: [x_260], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_260 => add_222, mul_356, mul_357, sub_102
# Graph fragment:
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_123, %unsqueeze_817), kwargs = {})
#   %mul_356 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %unsqueeze_819), kwargs = {})
#   %mul_357 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_356, %unsqueeze_821), kwargs = {})
#   %add_222 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_357, %unsqueeze_823), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_72 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_72', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_72', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_72(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/y6/cy6alsiyga2vjzzx6575ohun7dhk37xfb7jyzousymfjyuz4bi7z.py
# Topologically Sorted Source Nodes: [x_261], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_261 => cat_17
# Graph fragment:
#   %cat_17 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_223, %slice_72], 1), kwargs = {})
triton_poi_fused_cat_73 = async_compile.triton('triton_poi_fused_cat_73', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_73', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_73(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 117, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((128*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((117*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 128, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (117 + (128*x1) + ((-117) + x0)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5w/c5whjwsjaru336qvkgt5iy7swdhdovevp57tkoajiik3lr4kehzz.py
# Topologically Sorted Source Nodes: [x_263, x_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_263 => add_225, mul_359, mul_360, sub_103
#   x_264 => mul_361, sigmoid_49
# Graph fragment:
#   %sub_103 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_124, %unsqueeze_825), kwargs = {})
#   %mul_359 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_103, %unsqueeze_827), kwargs = {})
#   %mul_360 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_359, %unsqueeze_829), kwargs = {})
#   %add_225 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_360, %unsqueeze_831), kwargs = {})
#   %sigmoid_49 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_225,), kwargs = {})
#   %mul_361 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_225, %sigmoid_49), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_74 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_74', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_74', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_74(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 768
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ed/cedasl526ka4p4o5y3u3ujojljbmva2ftldyctwwisuoct4y5m3s.py
# Topologically Sorted Source Nodes: [x_266], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_266 => add_227, mul_363, mul_364, sub_104
# Graph fragment:
#   %sub_104 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_125, %unsqueeze_833), kwargs = {})
#   %mul_363 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_104, %unsqueeze_835), kwargs = {})
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_363, %unsqueeze_837), kwargs = {})
#   %add_227 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_364, %unsqueeze_839), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_75 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_75', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_75', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_75(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
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


# kernel path: /tmp/torchinductor_sahanp/l5/cl5swfaamx7w5nsw5izmky2jzwr4m7u6an5eu5x767smyhl5gilo.py
# Topologically Sorted Source Nodes: [x_se_84], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_84 => mean_22
# Graph fragment:
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_227, [2, 3], True), kwargs = {})
triton_per_fused_mean_76 = async_compile.triton('triton_per_fused_mean_76', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_76', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_76(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (37632*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4q/c4q3gtmtbsteh5ojdmzdiakiug7gvkutkp7n3i3x6h7bpsxj736s.py
# Topologically Sorted Source Nodes: [x_se_84, x_se_85, batch_norm_105, x_se_86], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_105 => add_229, mul_366, mul_367, sub_105
#   x_se_84 => mean_22
#   x_se_85 => convolution_126
#   x_se_86 => relu_21
# Graph fragment:
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_227, [2, 3], True), kwargs = {})
#   %convolution_126 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_22, %arg240_1, %arg241_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_126, %unsqueeze_841), kwargs = {})
#   %mul_366 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_367 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_366, %unsqueeze_845), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_367, %unsqueeze_847), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_229,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_77 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_77', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_77', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_77(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2d/c2dkhcf6zvvg7yhaxmrioz3gp4rsqxaslkflwvi5xpcuobs2invj.py
# Topologically Sorted Source Nodes: [x_se_84, x_se_85, batch_norm_105, x_se_86, x_se_87, sigmoid_21, x_267, x_268], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_105 => add_229, mul_366, mul_367, sub_105
#   sigmoid_21 => sigmoid_50
#   x_267 => mul_368
#   x_268 => clamp_max_27, clamp_min_27
#   x_se_84 => mean_22
#   x_se_85 => convolution_126
#   x_se_86 => relu_21
#   x_se_87 => convolution_127
# Graph fragment:
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_227, [2, 3], True), kwargs = {})
#   %convolution_126 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_22, %arg240_1, %arg241_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_126, %unsqueeze_841), kwargs = {})
#   %mul_366 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_367 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_366, %unsqueeze_845), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_367, %unsqueeze_847), kwargs = {})
#   %relu_21 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_229,), kwargs = {})
#   %convolution_127 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg246_1, %arg247_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_50 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_127,), kwargs = {})
#   %mul_368 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_227, %sigmoid_50), kwargs = {})
#   %clamp_min_27 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_368, 0.0), kwargs = {})
#   %clamp_max_27 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_27, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_78 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_78', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_78', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_78(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 768
    x2 = (xindex // 37632)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (768*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3r/c3rjhb77rizzfquugy5t5qfflnsktyzblrl22lhgdw5dmzzz75zs.py
# Topologically Sorted Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_270 => add_231, mul_370, mul_371, sub_106
# Graph fragment:
#   %sub_106 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_128, %unsqueeze_849), kwargs = {})
#   %mul_370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_106, %unsqueeze_851), kwargs = {})
#   %mul_371 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_370, %unsqueeze_853), kwargs = {})
#   %add_231 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_371, %unsqueeze_855), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_79 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_79', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_79', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_79(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 54880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 140
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


# kernel path: /tmp/torchinductor_sahanp/tn/ctnqkk55qvtxb4y4hv2qdl7ndm5yw55ctzp4fsomh62ekdtu77ve.py
# Topologically Sorted Source Nodes: [x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_272 => add_233, mul_373, mul_374, sub_107
#   x_273 => mul_375, sigmoid_51
# Graph fragment:
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_129, %unsqueeze_857), kwargs = {})
#   %mul_373 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %unsqueeze_859), kwargs = {})
#   %mul_374 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_373, %unsqueeze_861), kwargs = {})
#   %add_233 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_374, %unsqueeze_863), kwargs = {})
#   %sigmoid_51 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_233,), kwargs = {})
#   %mul_375 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_233, %sigmoid_51), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_80 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_80', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_80', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_80(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 329280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 840
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/v7/cv74cbxhm2j76sfrwuzuzqr5wadc33lbpezvqtxn4ttgaw6ltqz2.py
# Topologically Sorted Source Nodes: [x_275], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_275 => add_235, mul_377, mul_378, sub_108
# Graph fragment:
#   %sub_108 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_130, %unsqueeze_865), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_108, %unsqueeze_867), kwargs = {})
#   %mul_378 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_377, %unsqueeze_869), kwargs = {})
#   %add_235 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_378, %unsqueeze_871), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_81 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_81', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_81', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_81(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 329280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 840
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


# kernel path: /tmp/torchinductor_sahanp/vv/cvv4uflvko24iom6toyzeso7njyh3bxtlnuu7hflb5cs324hmrtw.py
# Topologically Sorted Source Nodes: [x_se_88], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_88 => mean_23
# Graph fragment:
#   %mean_23 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_235, [2, 3], True), kwargs = {})
triton_per_fused_mean_82 = async_compile.triton('triton_per_fused_mean_82', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_82', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_82(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6720
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 840
    x1 = (xindex // 840)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (840*r2) + (41160*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ee/cee7qlmmz7l33qoarxooxwk3q7542uwqjaldbvkycctjmyhh5dl5.py
# Topologically Sorted Source Nodes: [x_se_88, x_se_89, batch_norm_109, x_se_90], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_109 => add_237, mul_380, mul_381, sub_109
#   x_se_88 => mean_23
#   x_se_89 => convolution_131
#   x_se_90 => relu_22
# Graph fragment:
#   %mean_23 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_235, [2, 3], True), kwargs = {})
#   %convolution_131 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_23, %arg263_1, %arg264_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_109 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_131, %unsqueeze_873), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_109, %unsqueeze_875), kwargs = {})
#   %mul_381 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_380, %unsqueeze_877), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_381, %unsqueeze_879), kwargs = {})
#   %relu_22 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_237,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_83 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_83', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_83', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_83(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 70
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pt/cptqxyavcphy2or4bttxvuso65inpskz5eehghzwq5q7ffv6a5rg.py
# Topologically Sorted Source Nodes: [x_se_88, x_se_89, batch_norm_109, x_se_90, x_se_91, sigmoid_22, x_276, x_277], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_109 => add_237, mul_380, mul_381, sub_109
#   sigmoid_22 => sigmoid_52
#   x_276 => mul_382
#   x_277 => clamp_max_28, clamp_min_28
#   x_se_88 => mean_23
#   x_se_89 => convolution_131
#   x_se_90 => relu_22
#   x_se_91 => convolution_132
# Graph fragment:
#   %mean_23 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_235, [2, 3], True), kwargs = {})
#   %convolution_131 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_23, %arg263_1, %arg264_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_109 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_131, %unsqueeze_873), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_109, %unsqueeze_875), kwargs = {})
#   %mul_381 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_380, %unsqueeze_877), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_381, %unsqueeze_879), kwargs = {})
#   %relu_22 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_237,), kwargs = {})
#   %convolution_132 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %arg269_1, %arg270_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_52 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_132,), kwargs = {})
#   %mul_382 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_235, %sigmoid_52), kwargs = {})
#   %clamp_min_28 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_382, 0.0), kwargs = {})
#   %clamp_max_28 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_28, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_84 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_84', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_84', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_84(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 329280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 840
    x2 = (xindex // 41160)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (840*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/r5/cr5mewy3kosekrufs3mjn4h2j7763yvmpotgpmobfygdi5yfl3gf.py
# Topologically Sorted Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_279 => add_239, mul_384, mul_385, sub_110
# Graph fragment:
#   %sub_110 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_133, %unsqueeze_881), kwargs = {})
#   %mul_384 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_110, %unsqueeze_883), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_384, %unsqueeze_885), kwargs = {})
#   %add_239 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_385, %unsqueeze_887), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_85 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_85', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_85', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_85(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 59192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 151
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


# kernel path: /tmp/torchinductor_sahanp/yi/cyihj5j2d2hpf4foxfr2gtwpaesof6mr2r7xwaovvfnx35h3hl73.py
# Topologically Sorted Source Nodes: [x_280], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_280 => cat_18
# Graph fragment:
#   %cat_18 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_240, %slice_76], 1), kwargs = {})
triton_poi_fused_cat_86 = async_compile.triton('triton_poi_fused_cat_86', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_86', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_86(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 59192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 151
    x1 = (xindex // 151)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 140, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((151*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((140*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 151, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (140 + (151*x1) + ((-140) + x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g5/cg57tf7uishgimqtqkqyah7b6z65xb6fncfbenvdluzeiudzwzsd.py
# Topologically Sorted Source Nodes: [x_282, x_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_282 => add_242, mul_387, mul_388, sub_111
#   x_283 => mul_389, sigmoid_53
# Graph fragment:
#   %sub_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_134, %unsqueeze_889), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_111, %unsqueeze_891), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_387, %unsqueeze_893), kwargs = {})
#   %add_242 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_388, %unsqueeze_895), kwargs = {})
#   %sigmoid_53 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_242,), kwargs = {})
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_242, %sigmoid_53), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_87 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_87', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_87', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_87(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 355152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 906
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/z6/cz65v3we2nmlzvnei6e2flx4ed2ortiqfgtyzahqgb35zkjrjdh2.py
# Topologically Sorted Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_285 => add_244, mul_391, mul_392, sub_112
# Graph fragment:
#   %sub_112 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_135, %unsqueeze_897), kwargs = {})
#   %mul_391 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_112, %unsqueeze_899), kwargs = {})
#   %mul_392 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_391, %unsqueeze_901), kwargs = {})
#   %add_244 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_392, %unsqueeze_903), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_88 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_88', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_88', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_88(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 355152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 906
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


# kernel path: /tmp/torchinductor_sahanp/ht/cht66ru5kdc3gqvi4vfy5uyyybiowxde76qxquiw47n6byzj5iw5.py
# Topologically Sorted Source Nodes: [x_se_92], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_92 => mean_24
# Graph fragment:
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_244, [2, 3], True), kwargs = {})
triton_per_fused_mean_89 = async_compile.triton('triton_per_fused_mean_89', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_89', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_89(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7248
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 906
    x1 = (xindex // 906)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (906*r2) + (44394*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6r/c6r4pybkii55ahotztejam2plv5c2eqvqf4y7tn7lk4h63sgbgwx.py
# Topologically Sorted Source Nodes: [x_se_92, x_se_93, batch_norm_113, x_se_94], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_113 => add_246, mul_394, mul_395, sub_113
#   x_se_92 => mean_24
#   x_se_93 => convolution_136
#   x_se_94 => relu_23
# Graph fragment:
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_244, [2, 3], True), kwargs = {})
#   %convolution_136 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_24, %arg286_1, %arg287_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_136, %unsqueeze_905), kwargs = {})
#   %mul_394 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %unsqueeze_907), kwargs = {})
#   %mul_395 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_394, %unsqueeze_909), kwargs = {})
#   %add_246 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_395, %unsqueeze_911), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_246,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_90 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_90', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_90', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_90(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 75
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kh/ckhf5wbip6z2h7hy66fjdrdk23dfjmo6a6ae6eo7dvju3p6wv4xb.py
# Topologically Sorted Source Nodes: [x_se_92, x_se_93, batch_norm_113, x_se_94, x_se_95, sigmoid_23, x_286, x_287], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_113 => add_246, mul_394, mul_395, sub_113
#   sigmoid_23 => sigmoid_54
#   x_286 => mul_396
#   x_287 => clamp_max_29, clamp_min_29
#   x_se_92 => mean_24
#   x_se_93 => convolution_136
#   x_se_94 => relu_23
#   x_se_95 => convolution_137
# Graph fragment:
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_244, [2, 3], True), kwargs = {})
#   %convolution_136 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_24, %arg286_1, %arg287_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_136, %unsqueeze_905), kwargs = {})
#   %mul_394 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %unsqueeze_907), kwargs = {})
#   %mul_395 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_394, %unsqueeze_909), kwargs = {})
#   %add_246 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_395, %unsqueeze_911), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_246,), kwargs = {})
#   %convolution_137 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg292_1, %arg293_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_54 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_137,), kwargs = {})
#   %mul_396 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_244, %sigmoid_54), kwargs = {})
#   %clamp_min_29 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_396, 0.0), kwargs = {})
#   %clamp_max_29 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_29, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_91 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_91', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_91', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_91(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 355152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 906
    x2 = (xindex // 44394)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (906*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hk/chkymvcdmabv62c5trzkqyihngdldqwoorw3de4pewn7iqa5a4yd.py
# Topologically Sorted Source Nodes: [x_289], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_289 => add_248, mul_398, mul_399, sub_114
# Graph fragment:
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_138, %unsqueeze_913), kwargs = {})
#   %mul_398 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_114, %unsqueeze_915), kwargs = {})
#   %mul_399 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_398, %unsqueeze_917), kwargs = {})
#   %add_248 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_399, %unsqueeze_919), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_92 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_92', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_92', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_92(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 63504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 162
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


# kernel path: /tmp/torchinductor_sahanp/xi/cxi5avdvgkk4ki6tmpc2wtdxge3zmzvbjdtqzwkojxblnarwwu3c.py
# Topologically Sorted Source Nodes: [x_290], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_290 => cat_19
# Graph fragment:
#   %cat_19 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_249, %slice_80], 1), kwargs = {})
triton_poi_fused_cat_93 = async_compile.triton('triton_poi_fused_cat_93', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_93', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_93(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 63504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 162
    x1 = (xindex // 162)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 151, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((162*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((151*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 162, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (151 + (162*x1) + ((-151) + x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/az/cazku2eueeqr42aryhuwdm3syfpxxjfwkuq37xqujtdj7nh6le6t.py
# Topologically Sorted Source Nodes: [x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_292 => add_251, mul_401, mul_402, sub_115
#   x_293 => mul_403, sigmoid_55
# Graph fragment:
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_139, %unsqueeze_921), kwargs = {})
#   %mul_401 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %unsqueeze_923), kwargs = {})
#   %mul_402 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_401, %unsqueeze_925), kwargs = {})
#   %add_251 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_402, %unsqueeze_927), kwargs = {})
#   %sigmoid_55 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_251,), kwargs = {})
#   %mul_403 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_251, %sigmoid_55), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_94 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_94', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_94', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_94(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 381024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 972
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ca/ccavjreqsu2ffegl4pcbwtooeg7ir2ypu7bg6dslx5a3ldxhknsa.py
# Topologically Sorted Source Nodes: [x_295], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_295 => add_253, mul_405, mul_406, sub_116
# Graph fragment:
#   %sub_116 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_140, %unsqueeze_929), kwargs = {})
#   %mul_405 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_116, %unsqueeze_931), kwargs = {})
#   %mul_406 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_405, %unsqueeze_933), kwargs = {})
#   %add_253 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_406, %unsqueeze_935), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_95 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_95', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_95', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_95(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 381024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 972
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


# kernel path: /tmp/torchinductor_sahanp/oh/coh43fmc5gcl6tsx6e3n55th6rgzg2psc4355jrgawxl7mwpdb4m.py
# Topologically Sorted Source Nodes: [x_se_96], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_96 => mean_25
# Graph fragment:
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_253, [2, 3], True), kwargs = {})
triton_per_fused_mean_96 = async_compile.triton('triton_per_fused_mean_96', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_96', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_96(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7776
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 972
    x1 = (xindex // 972)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (972*r2) + (47628*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ef/ceflxq32phjo5w3zg3rx66fmmylvgdbjhfn4fcfo5ivh546hilci.py
# Topologically Sorted Source Nodes: [x_se_96, x_se_97, batch_norm_117, x_se_98], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_117 => add_255, mul_408, mul_409, sub_117
#   x_se_96 => mean_25
#   x_se_97 => convolution_141
#   x_se_98 => relu_24
# Graph fragment:
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_253, [2, 3], True), kwargs = {})
#   %convolution_141 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_25, %arg309_1, %arg310_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_117 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_141, %unsqueeze_937), kwargs = {})
#   %mul_408 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_117, %unsqueeze_939), kwargs = {})
#   %mul_409 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_408, %unsqueeze_941), kwargs = {})
#   %add_255 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_409, %unsqueeze_943), kwargs = {})
#   %relu_24 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_255,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_97 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_97', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_97', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_97(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 81
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ui/cuio3jafsbea7iifq2otlimpsvmsfuwoput7y23qqsplp2cf3nix.py
# Topologically Sorted Source Nodes: [x_se_96, x_se_97, batch_norm_117, x_se_98, x_se_99, sigmoid_24, x_296, x_297], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_117 => add_255, mul_408, mul_409, sub_117
#   sigmoid_24 => sigmoid_56
#   x_296 => mul_410
#   x_297 => clamp_max_30, clamp_min_30
#   x_se_96 => mean_25
#   x_se_97 => convolution_141
#   x_se_98 => relu_24
#   x_se_99 => convolution_142
# Graph fragment:
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_253, [2, 3], True), kwargs = {})
#   %convolution_141 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_25, %arg309_1, %arg310_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_117 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_141, %unsqueeze_937), kwargs = {})
#   %mul_408 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_117, %unsqueeze_939), kwargs = {})
#   %mul_409 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_408, %unsqueeze_941), kwargs = {})
#   %add_255 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_409, %unsqueeze_943), kwargs = {})
#   %relu_24 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_255,), kwargs = {})
#   %convolution_142 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %arg315_1, %arg316_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_56 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_142,), kwargs = {})
#   %mul_410 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_253, %sigmoid_56), kwargs = {})
#   %clamp_min_30 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_410, 0.0), kwargs = {})
#   %clamp_max_30 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_30, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_98 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_98', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_98', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_98(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 381024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 972
    x2 = (xindex // 47628)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (972*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g7/cg7wa7a43iai3dbyp5l76gsvnxqzbetw2xmkat44flfdr2corzy6.py
# Topologically Sorted Source Nodes: [x_299], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_299 => add_257, mul_412, mul_413, sub_118
# Graph fragment:
#   %sub_118 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_143, %unsqueeze_945), kwargs = {})
#   %mul_412 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_118, %unsqueeze_947), kwargs = {})
#   %mul_413 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_412, %unsqueeze_949), kwargs = {})
#   %add_257 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_413, %unsqueeze_951), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_99 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_99', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_99', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_99(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 68208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 174
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


# kernel path: /tmp/torchinductor_sahanp/px/cpxvi5lyu7viixr5ct4rhxrutjdlalocpk6efsifkvbmjxvxcg2m.py
# Topologically Sorted Source Nodes: [x_300], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_300 => cat_20
# Graph fragment:
#   %cat_20 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_258, %slice_84], 1), kwargs = {})
triton_poi_fused_cat_100 = async_compile.triton('triton_poi_fused_cat_100', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_100', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_100(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 68208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 174
    x1 = (xindex // 174)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 162, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((174*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((162*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 174, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (162 + (174*x1) + ((-162) + x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5s/c5sdoslieo6fysx7usa5c5pzl7sqrgcm24dpwknuenkiksva7czq.py
# Topologically Sorted Source Nodes: [x_302, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_302 => add_260, mul_415, mul_416, sub_119
#   x_303 => mul_417, sigmoid_57
# Graph fragment:
#   %sub_119 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_144, %unsqueeze_953), kwargs = {})
#   %mul_415 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_119, %unsqueeze_955), kwargs = {})
#   %mul_416 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_415, %unsqueeze_957), kwargs = {})
#   %add_260 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_416, %unsqueeze_959), kwargs = {})
#   %sigmoid_57 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_260,), kwargs = {})
#   %mul_417 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_260, %sigmoid_57), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_101 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_101', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_101', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_101(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1044
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fx/cfxe62x2dktfk3ny3o44khiifoxeqmf7jg7loinfpwsyev724pyt.py
# Topologically Sorted Source Nodes: [x_305], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_305 => add_262, mul_419, mul_420, sub_120
# Graph fragment:
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_145, %unsqueeze_961), kwargs = {})
#   %mul_419 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_120, %unsqueeze_963), kwargs = {})
#   %mul_420 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_419, %unsqueeze_965), kwargs = {})
#   %add_262 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_420, %unsqueeze_967), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_102 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_102', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_102', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_102(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1044
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


# kernel path: /tmp/torchinductor_sahanp/7k/c7kaih56o4l6ocjx7k4luf6fmslt3wefeck4fvstvveukiszcj4m.py
# Topologically Sorted Source Nodes: [x_se_100], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_100 => mean_26
# Graph fragment:
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_262, [2, 3], True), kwargs = {})
triton_per_fused_mean_103 = async_compile.triton('triton_per_fused_mean_103', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_103', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_103(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1044
    x1 = (xindex // 1044)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1044*r2) + (51156*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sm/csm6xhvdytz4tdvkg5dr24gi6ncyhgmmf7wls2bagkx4eo5jwj5w.py
# Topologically Sorted Source Nodes: [x_se_100, x_se_101, batch_norm_121, x_se_102], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_121 => add_264, mul_422, mul_423, sub_121
#   x_se_100 => mean_26
#   x_se_101 => convolution_146
#   x_se_102 => relu_25
# Graph fragment:
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_262, [2, 3], True), kwargs = {})
#   %convolution_146 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_26, %arg332_1, %arg333_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_121 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_146, %unsqueeze_969), kwargs = {})
#   %mul_422 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_121, %unsqueeze_971), kwargs = {})
#   %mul_423 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_422, %unsqueeze_973), kwargs = {})
#   %add_264 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_423, %unsqueeze_975), kwargs = {})
#   %relu_25 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_264,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_104 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_104', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_104', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_104(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 87
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uv/cuvlwjz2p2om4dxlq336mdm6eb6nuoxdynjkqmxwj3byhs732nvf.py
# Topologically Sorted Source Nodes: [x_se_100, x_se_101, batch_norm_121, x_se_102, x_se_103, sigmoid_25, x_306, x_307], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
# Source node to ATen node mapping:
#   batch_norm_121 => add_264, mul_422, mul_423, sub_121
#   sigmoid_25 => sigmoid_58
#   x_306 => mul_424
#   x_307 => clamp_max_31, clamp_min_31
#   x_se_100 => mean_26
#   x_se_101 => convolution_146
#   x_se_102 => relu_25
#   x_se_103 => convolution_147
# Graph fragment:
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_262, [2, 3], True), kwargs = {})
#   %convolution_146 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_26, %arg332_1, %arg333_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_121 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_146, %unsqueeze_969), kwargs = {})
#   %mul_422 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_121, %unsqueeze_971), kwargs = {})
#   %mul_423 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_422, %unsqueeze_973), kwargs = {})
#   %add_264 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_423, %unsqueeze_975), kwargs = {})
#   %relu_25 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_264,), kwargs = {})
#   %convolution_147 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_25, %arg338_1, %arg339_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_58 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_147,), kwargs = {})
#   %mul_424 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_262, %sigmoid_58), kwargs = {})
#   %clamp_min_31 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_424, 0.0), kwargs = {})
#   %clamp_max_31 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_31, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_105 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_105', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_105', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_105(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1044
    x2 = (xindex // 51156)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (1044*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp0 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fc/cfchp5khkkaorxqt3f3uhb5u6ag2ynb2t6etktdv3ts67iatzooh.py
# Topologically Sorted Source Nodes: [x_309], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_309 => add_266, mul_426, mul_427, sub_122
# Graph fragment:
#   %sub_122 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_148, %unsqueeze_977), kwargs = {})
#   %mul_426 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_122, %unsqueeze_979), kwargs = {})
#   %mul_427 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_426, %unsqueeze_981), kwargs = {})
#   %add_266 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_427, %unsqueeze_983), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_106 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_106', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_106', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_106(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 185
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


# kernel path: /tmp/torchinductor_sahanp/ok/cokmatwt64hkl2r5zequ7tvkxqrknifzz3mnaf5mywyvjxvylslk.py
# Topologically Sorted Source Nodes: [x_310], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_310 => cat_21
# Graph fragment:
#   %cat_21 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_267, %slice_88], 1), kwargs = {})
triton_poi_fused_cat_107 = async_compile.triton('triton_poi_fused_cat_107', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_107', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_107(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 185
    x1 = (xindex // 185)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 174, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((185*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((174*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 185, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (174 + (185*x1) + ((-174) + x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/p3/cp3n3ucdlqv5jth54i5ryzkd3suov6hpi4qbrr7jn2wvb3qqguop.py
# Topologically Sorted Source Nodes: [x_312], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_312 => add_269, mul_429, mul_430, sub_123
# Graph fragment:
#   %sub_123 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_149, %unsqueeze_985), kwargs = {})
#   %mul_429 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_123, %unsqueeze_987), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_429, %unsqueeze_989), kwargs = {})
#   %add_269 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_430, %unsqueeze_991), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_108 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_108', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_108', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_108(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
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


# kernel path: /tmp/torchinductor_sahanp/av/cav6incuh7i347huj6ehdrpgqe6235pmvax7pd3zkkgzavphml3t.py
# Topologically Sorted Source Nodes: [x_313, x_314], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_313 => mul_431, sigmoid_59
#   x_314 => mean_27
# Graph fragment:
#   %sigmoid_59 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_269,), kwargs = {})
#   %mul_431 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_269, %sigmoid_59), kwargs = {})
#   %mean_27 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_431, [-1, -2], True), kwargs = {})
triton_per_fused_mean_silu_109 = async_compile.triton('triton_per_fused_mean_silu_109', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_109', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_109(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1 = args
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
    assert_size_stride(arg26_1, (27, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg27_1, (27, ), (1, ))
    assert_size_stride(arg28_1, (27, ), (1, ))
    assert_size_stride(arg29_1, (27, ), (1, ))
    assert_size_stride(arg30_1, (27, ), (1, ))
    assert_size_stride(arg31_1, (162, 27, 1, 1), (27, 1, 1, 1))
    assert_size_stride(arg32_1, (162, ), (1, ))
    assert_size_stride(arg33_1, (162, ), (1, ))
    assert_size_stride(arg34_1, (162, ), (1, ))
    assert_size_stride(arg35_1, (162, ), (1, ))
    assert_size_stride(arg36_1, (162, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg37_1, (162, ), (1, ))
    assert_size_stride(arg38_1, (162, ), (1, ))
    assert_size_stride(arg39_1, (162, ), (1, ))
    assert_size_stride(arg40_1, (162, ), (1, ))
    assert_size_stride(arg41_1, (38, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(arg42_1, (38, ), (1, ))
    assert_size_stride(arg43_1, (38, ), (1, ))
    assert_size_stride(arg44_1, (38, ), (1, ))
    assert_size_stride(arg45_1, (38, ), (1, ))
    assert_size_stride(arg46_1, (228, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(arg47_1, (228, ), (1, ))
    assert_size_stride(arg48_1, (228, ), (1, ))
    assert_size_stride(arg49_1, (228, ), (1, ))
    assert_size_stride(arg50_1, (228, ), (1, ))
    assert_size_stride(arg51_1, (228, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg52_1, (228, ), (1, ))
    assert_size_stride(arg53_1, (228, ), (1, ))
    assert_size_stride(arg54_1, (228, ), (1, ))
    assert_size_stride(arg55_1, (228, ), (1, ))
    assert_size_stride(arg56_1, (19, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(arg57_1, (19, ), (1, ))
    assert_size_stride(arg58_1, (19, ), (1, ))
    assert_size_stride(arg59_1, (19, ), (1, ))
    assert_size_stride(arg60_1, (19, ), (1, ))
    assert_size_stride(arg61_1, (19, ), (1, ))
    assert_size_stride(arg62_1, (228, 19, 1, 1), (19, 1, 1, 1))
    assert_size_stride(arg63_1, (228, ), (1, ))
    assert_size_stride(arg64_1, (50, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(arg65_1, (50, ), (1, ))
    assert_size_stride(arg66_1, (50, ), (1, ))
    assert_size_stride(arg67_1, (50, ), (1, ))
    assert_size_stride(arg68_1, (50, ), (1, ))
    assert_size_stride(arg69_1, (300, 50, 1, 1), (50, 1, 1, 1))
    assert_size_stride(arg70_1, (300, ), (1, ))
    assert_size_stride(arg71_1, (300, ), (1, ))
    assert_size_stride(arg72_1, (300, ), (1, ))
    assert_size_stride(arg73_1, (300, ), (1, ))
    assert_size_stride(arg74_1, (300, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg75_1, (300, ), (1, ))
    assert_size_stride(arg76_1, (300, ), (1, ))
    assert_size_stride(arg77_1, (300, ), (1, ))
    assert_size_stride(arg78_1, (300, ), (1, ))
    assert_size_stride(arg79_1, (25, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(arg80_1, (25, ), (1, ))
    assert_size_stride(arg81_1, (25, ), (1, ))
    assert_size_stride(arg82_1, (25, ), (1, ))
    assert_size_stride(arg83_1, (25, ), (1, ))
    assert_size_stride(arg84_1, (25, ), (1, ))
    assert_size_stride(arg85_1, (300, 25, 1, 1), (25, 1, 1, 1))
    assert_size_stride(arg86_1, (300, ), (1, ))
    assert_size_stride(arg87_1, (61, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(arg88_1, (61, ), (1, ))
    assert_size_stride(arg89_1, (61, ), (1, ))
    assert_size_stride(arg90_1, (61, ), (1, ))
    assert_size_stride(arg91_1, (61, ), (1, ))
    assert_size_stride(arg92_1, (366, 61, 1, 1), (61, 1, 1, 1))
    assert_size_stride(arg93_1, (366, ), (1, ))
    assert_size_stride(arg94_1, (366, ), (1, ))
    assert_size_stride(arg95_1, (366, ), (1, ))
    assert_size_stride(arg96_1, (366, ), (1, ))
    assert_size_stride(arg97_1, (366, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg98_1, (366, ), (1, ))
    assert_size_stride(arg99_1, (366, ), (1, ))
    assert_size_stride(arg100_1, (366, ), (1, ))
    assert_size_stride(arg101_1, (366, ), (1, ))
    assert_size_stride(arg102_1, (30, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(arg103_1, (30, ), (1, ))
    assert_size_stride(arg104_1, (30, ), (1, ))
    assert_size_stride(arg105_1, (30, ), (1, ))
    assert_size_stride(arg106_1, (30, ), (1, ))
    assert_size_stride(arg107_1, (30, ), (1, ))
    assert_size_stride(arg108_1, (366, 30, 1, 1), (30, 1, 1, 1))
    assert_size_stride(arg109_1, (366, ), (1, ))
    assert_size_stride(arg110_1, (72, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(arg111_1, (72, ), (1, ))
    assert_size_stride(arg112_1, (72, ), (1, ))
    assert_size_stride(arg113_1, (72, ), (1, ))
    assert_size_stride(arg114_1, (72, ), (1, ))
    assert_size_stride(arg115_1, (432, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg116_1, (432, ), (1, ))
    assert_size_stride(arg117_1, (432, ), (1, ))
    assert_size_stride(arg118_1, (432, ), (1, ))
    assert_size_stride(arg119_1, (432, ), (1, ))
    assert_size_stride(arg120_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg121_1, (432, ), (1, ))
    assert_size_stride(arg122_1, (432, ), (1, ))
    assert_size_stride(arg123_1, (432, ), (1, ))
    assert_size_stride(arg124_1, (432, ), (1, ))
    assert_size_stride(arg125_1, (36, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg126_1, (36, ), (1, ))
    assert_size_stride(arg127_1, (36, ), (1, ))
    assert_size_stride(arg128_1, (36, ), (1, ))
    assert_size_stride(arg129_1, (36, ), (1, ))
    assert_size_stride(arg130_1, (36, ), (1, ))
    assert_size_stride(arg131_1, (432, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(arg132_1, (432, ), (1, ))
    assert_size_stride(arg133_1, (84, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg134_1, (84, ), (1, ))
    assert_size_stride(arg135_1, (84, ), (1, ))
    assert_size_stride(arg136_1, (84, ), (1, ))
    assert_size_stride(arg137_1, (84, ), (1, ))
    assert_size_stride(arg138_1, (504, 84, 1, 1), (84, 1, 1, 1))
    assert_size_stride(arg139_1, (504, ), (1, ))
    assert_size_stride(arg140_1, (504, ), (1, ))
    assert_size_stride(arg141_1, (504, ), (1, ))
    assert_size_stride(arg142_1, (504, ), (1, ))
    assert_size_stride(arg143_1, (504, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg144_1, (504, ), (1, ))
    assert_size_stride(arg145_1, (504, ), (1, ))
    assert_size_stride(arg146_1, (504, ), (1, ))
    assert_size_stride(arg147_1, (504, ), (1, ))
    assert_size_stride(arg148_1, (42, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(arg149_1, (42, ), (1, ))
    assert_size_stride(arg150_1, (42, ), (1, ))
    assert_size_stride(arg151_1, (42, ), (1, ))
    assert_size_stride(arg152_1, (42, ), (1, ))
    assert_size_stride(arg153_1, (42, ), (1, ))
    assert_size_stride(arg154_1, (504, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(arg155_1, (504, ), (1, ))
    assert_size_stride(arg156_1, (95, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(arg157_1, (95, ), (1, ))
    assert_size_stride(arg158_1, (95, ), (1, ))
    assert_size_stride(arg159_1, (95, ), (1, ))
    assert_size_stride(arg160_1, (95, ), (1, ))
    assert_size_stride(arg161_1, (570, 95, 1, 1), (95, 1, 1, 1))
    assert_size_stride(arg162_1, (570, ), (1, ))
    assert_size_stride(arg163_1, (570, ), (1, ))
    assert_size_stride(arg164_1, (570, ), (1, ))
    assert_size_stride(arg165_1, (570, ), (1, ))
    assert_size_stride(arg166_1, (570, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg167_1, (570, ), (1, ))
    assert_size_stride(arg168_1, (570, ), (1, ))
    assert_size_stride(arg169_1, (570, ), (1, ))
    assert_size_stride(arg170_1, (570, ), (1, ))
    assert_size_stride(arg171_1, (47, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(arg172_1, (47, ), (1, ))
    assert_size_stride(arg173_1, (47, ), (1, ))
    assert_size_stride(arg174_1, (47, ), (1, ))
    assert_size_stride(arg175_1, (47, ), (1, ))
    assert_size_stride(arg176_1, (47, ), (1, ))
    assert_size_stride(arg177_1, (570, 47, 1, 1), (47, 1, 1, 1))
    assert_size_stride(arg178_1, (570, ), (1, ))
    assert_size_stride(arg179_1, (106, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(arg180_1, (106, ), (1, ))
    assert_size_stride(arg181_1, (106, ), (1, ))
    assert_size_stride(arg182_1, (106, ), (1, ))
    assert_size_stride(arg183_1, (106, ), (1, ))
    assert_size_stride(arg184_1, (636, 106, 1, 1), (106, 1, 1, 1))
    assert_size_stride(arg185_1, (636, ), (1, ))
    assert_size_stride(arg186_1, (636, ), (1, ))
    assert_size_stride(arg187_1, (636, ), (1, ))
    assert_size_stride(arg188_1, (636, ), (1, ))
    assert_size_stride(arg189_1, (636, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg190_1, (636, ), (1, ))
    assert_size_stride(arg191_1, (636, ), (1, ))
    assert_size_stride(arg192_1, (636, ), (1, ))
    assert_size_stride(arg193_1, (636, ), (1, ))
    assert_size_stride(arg194_1, (53, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(arg195_1, (53, ), (1, ))
    assert_size_stride(arg196_1, (53, ), (1, ))
    assert_size_stride(arg197_1, (53, ), (1, ))
    assert_size_stride(arg198_1, (53, ), (1, ))
    assert_size_stride(arg199_1, (53, ), (1, ))
    assert_size_stride(arg200_1, (636, 53, 1, 1), (53, 1, 1, 1))
    assert_size_stride(arg201_1, (636, ), (1, ))
    assert_size_stride(arg202_1, (117, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(arg203_1, (117, ), (1, ))
    assert_size_stride(arg204_1, (117, ), (1, ))
    assert_size_stride(arg205_1, (117, ), (1, ))
    assert_size_stride(arg206_1, (117, ), (1, ))
    assert_size_stride(arg207_1, (702, 117, 1, 1), (117, 1, 1, 1))
    assert_size_stride(arg208_1, (702, ), (1, ))
    assert_size_stride(arg209_1, (702, ), (1, ))
    assert_size_stride(arg210_1, (702, ), (1, ))
    assert_size_stride(arg211_1, (702, ), (1, ))
    assert_size_stride(arg212_1, (702, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg213_1, (702, ), (1, ))
    assert_size_stride(arg214_1, (702, ), (1, ))
    assert_size_stride(arg215_1, (702, ), (1, ))
    assert_size_stride(arg216_1, (702, ), (1, ))
    assert_size_stride(arg217_1, (58, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(arg218_1, (58, ), (1, ))
    assert_size_stride(arg219_1, (58, ), (1, ))
    assert_size_stride(arg220_1, (58, ), (1, ))
    assert_size_stride(arg221_1, (58, ), (1, ))
    assert_size_stride(arg222_1, (58, ), (1, ))
    assert_size_stride(arg223_1, (702, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg224_1, (702, ), (1, ))
    assert_size_stride(arg225_1, (128, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(arg226_1, (128, ), (1, ))
    assert_size_stride(arg227_1, (128, ), (1, ))
    assert_size_stride(arg228_1, (128, ), (1, ))
    assert_size_stride(arg229_1, (128, ), (1, ))
    assert_size_stride(arg230_1, (768, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg231_1, (768, ), (1, ))
    assert_size_stride(arg232_1, (768, ), (1, ))
    assert_size_stride(arg233_1, (768, ), (1, ))
    assert_size_stride(arg234_1, (768, ), (1, ))
    assert_size_stride(arg235_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg236_1, (768, ), (1, ))
    assert_size_stride(arg237_1, (768, ), (1, ))
    assert_size_stride(arg238_1, (768, ), (1, ))
    assert_size_stride(arg239_1, (768, ), (1, ))
    assert_size_stride(arg240_1, (64, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg241_1, (64, ), (1, ))
    assert_size_stride(arg242_1, (64, ), (1, ))
    assert_size_stride(arg243_1, (64, ), (1, ))
    assert_size_stride(arg244_1, (64, ), (1, ))
    assert_size_stride(arg245_1, (64, ), (1, ))
    assert_size_stride(arg246_1, (768, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg247_1, (768, ), (1, ))
    assert_size_stride(arg248_1, (140, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg249_1, (140, ), (1, ))
    assert_size_stride(arg250_1, (140, ), (1, ))
    assert_size_stride(arg251_1, (140, ), (1, ))
    assert_size_stride(arg252_1, (140, ), (1, ))
    assert_size_stride(arg253_1, (840, 140, 1, 1), (140, 1, 1, 1))
    assert_size_stride(arg254_1, (840, ), (1, ))
    assert_size_stride(arg255_1, (840, ), (1, ))
    assert_size_stride(arg256_1, (840, ), (1, ))
    assert_size_stride(arg257_1, (840, ), (1, ))
    assert_size_stride(arg258_1, (840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg259_1, (840, ), (1, ))
    assert_size_stride(arg260_1, (840, ), (1, ))
    assert_size_stride(arg261_1, (840, ), (1, ))
    assert_size_stride(arg262_1, (840, ), (1, ))
    assert_size_stride(arg263_1, (70, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(arg264_1, (70, ), (1, ))
    assert_size_stride(arg265_1, (70, ), (1, ))
    assert_size_stride(arg266_1, (70, ), (1, ))
    assert_size_stride(arg267_1, (70, ), (1, ))
    assert_size_stride(arg268_1, (70, ), (1, ))
    assert_size_stride(arg269_1, (840, 70, 1, 1), (70, 1, 1, 1))
    assert_size_stride(arg270_1, (840, ), (1, ))
    assert_size_stride(arg271_1, (151, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(arg272_1, (151, ), (1, ))
    assert_size_stride(arg273_1, (151, ), (1, ))
    assert_size_stride(arg274_1, (151, ), (1, ))
    assert_size_stride(arg275_1, (151, ), (1, ))
    assert_size_stride(arg276_1, (906, 151, 1, 1), (151, 1, 1, 1))
    assert_size_stride(arg277_1, (906, ), (1, ))
    assert_size_stride(arg278_1, (906, ), (1, ))
    assert_size_stride(arg279_1, (906, ), (1, ))
    assert_size_stride(arg280_1, (906, ), (1, ))
    assert_size_stride(arg281_1, (906, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg282_1, (906, ), (1, ))
    assert_size_stride(arg283_1, (906, ), (1, ))
    assert_size_stride(arg284_1, (906, ), (1, ))
    assert_size_stride(arg285_1, (906, ), (1, ))
    assert_size_stride(arg286_1, (75, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(arg287_1, (75, ), (1, ))
    assert_size_stride(arg288_1, (75, ), (1, ))
    assert_size_stride(arg289_1, (75, ), (1, ))
    assert_size_stride(arg290_1, (75, ), (1, ))
    assert_size_stride(arg291_1, (75, ), (1, ))
    assert_size_stride(arg292_1, (906, 75, 1, 1), (75, 1, 1, 1))
    assert_size_stride(arg293_1, (906, ), (1, ))
    assert_size_stride(arg294_1, (162, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(arg295_1, (162, ), (1, ))
    assert_size_stride(arg296_1, (162, ), (1, ))
    assert_size_stride(arg297_1, (162, ), (1, ))
    assert_size_stride(arg298_1, (162, ), (1, ))
    assert_size_stride(arg299_1, (972, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(arg300_1, (972, ), (1, ))
    assert_size_stride(arg301_1, (972, ), (1, ))
    assert_size_stride(arg302_1, (972, ), (1, ))
    assert_size_stride(arg303_1, (972, ), (1, ))
    assert_size_stride(arg304_1, (972, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg305_1, (972, ), (1, ))
    assert_size_stride(arg306_1, (972, ), (1, ))
    assert_size_stride(arg307_1, (972, ), (1, ))
    assert_size_stride(arg308_1, (972, ), (1, ))
    assert_size_stride(arg309_1, (81, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(arg310_1, (81, ), (1, ))
    assert_size_stride(arg311_1, (81, ), (1, ))
    assert_size_stride(arg312_1, (81, ), (1, ))
    assert_size_stride(arg313_1, (81, ), (1, ))
    assert_size_stride(arg314_1, (81, ), (1, ))
    assert_size_stride(arg315_1, (972, 81, 1, 1), (81, 1, 1, 1))
    assert_size_stride(arg316_1, (972, ), (1, ))
    assert_size_stride(arg317_1, (174, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(arg318_1, (174, ), (1, ))
    assert_size_stride(arg319_1, (174, ), (1, ))
    assert_size_stride(arg320_1, (174, ), (1, ))
    assert_size_stride(arg321_1, (174, ), (1, ))
    assert_size_stride(arg322_1, (1044, 174, 1, 1), (174, 1, 1, 1))
    assert_size_stride(arg323_1, (1044, ), (1, ))
    assert_size_stride(arg324_1, (1044, ), (1, ))
    assert_size_stride(arg325_1, (1044, ), (1, ))
    assert_size_stride(arg326_1, (1044, ), (1, ))
    assert_size_stride(arg327_1, (1044, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg328_1, (1044, ), (1, ))
    assert_size_stride(arg329_1, (1044, ), (1, ))
    assert_size_stride(arg330_1, (1044, ), (1, ))
    assert_size_stride(arg331_1, (1044, ), (1, ))
    assert_size_stride(arg332_1, (87, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(arg333_1, (87, ), (1, ))
    assert_size_stride(arg334_1, (87, ), (1, ))
    assert_size_stride(arg335_1, (87, ), (1, ))
    assert_size_stride(arg336_1, (87, ), (1, ))
    assert_size_stride(arg337_1, (87, ), (1, ))
    assert_size_stride(arg338_1, (1044, 87, 1, 1), (87, 1, 1, 1))
    assert_size_stride(arg339_1, (1044, ), (1, ))
    assert_size_stride(arg340_1, (185, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(arg341_1, (185, ), (1, ))
    assert_size_stride(arg342_1, (185, ), (1, ))
    assert_size_stride(arg343_1, (185, ), (1, ))
    assert_size_stride(arg344_1, (185, ), (1, ))
    assert_size_stride(arg345_1, (1280, 185, 1, 1), (185, 1, 1, 1))
    assert_size_stride(arg346_1, (1280, ), (1, ))
    assert_size_stride(arg347_1, (1280, ), (1, ))
    assert_size_stride(arg348_1, (1280, ), (1, ))
    assert_size_stride(arg349_1, (1280, ), (1, ))
    assert_size_stride(arg350_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg351_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_159], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_159], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_159], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((8, 32, 112, 112), (401408, 1, 3584, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_160, x_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, buf4, 3211264, grid=grid(3211264), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf3
        # Topologically Sorted Source Nodes: [x_161, x_162], Original ATen: [aten.silu, aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf5, (8, 32, 112, 112), (401408, 1, 3584, 32))
        del arg6_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_163, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_3.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [x_163, x_164, x_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 16, 112, 112), (200704, 1, 1792, 16))
        del arg11_1
        del buf6
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf8, arg12_1, arg13_1, arg14_1, arg15_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        # Topologically Sorted Source Nodes: [x_166, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 96, 112, 112), (1204224, 1, 10752, 96))
        del arg16_1
        del buf8
        buf10 = buf9; del buf9  # reuse
        buf11 = empty_strided_cuda((8, 96, 112, 112), (1204224, 1, 10752, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_168, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_5.run(buf10, arg17_1, arg18_1, arg19_1, arg20_1, buf11, 9633792, grid=grid(9633792), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del buf10
        # Topologically Sorted Source Nodes: [x_169, x_170], Original ATen: [aten.silu, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg21_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf12, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del arg21_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_171, x_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6.run(buf13, arg22_1, arg23_1, arg24_1, arg25_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        # Topologically Sorted Source Nodes: [x_171, x_172, x_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 27, 56, 56), (84672, 1, 1512, 27))
        del arg26_1
        del buf13
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_174], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_7.run(buf15, arg27_1, arg28_1, arg29_1, arg30_1, 677376, grid=grid(677376), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        # Topologically Sorted Source Nodes: [x_175], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 162, 56, 56), (508032, 1, 9072, 162))
        del arg31_1
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided_cuda((8, 162, 56, 56), (508032, 1, 9072, 162), torch.float32)
        # Topologically Sorted Source Nodes: [x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_8.run(buf17, arg32_1, arg33_1, arg34_1, arg35_1, buf18, 4064256, grid=grid(4064256), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        del buf17
        # Topologically Sorted Source Nodes: [x_177, x_178], Original ATen: [aten.silu, aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg36_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=162, bias=None)
        assert_size_stride(buf19, (8, 162, 56, 56), (508032, 1, 9072, 162))
        del arg36_1
        del buf18
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_179, x_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf20, arg37_1, arg38_1, arg39_1, arg40_1, 4064256, grid=grid(4064256), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        # Topologically Sorted Source Nodes: [x_179, x_180, x_181], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh, aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 38, 56, 56), (119168, 1, 2128, 38))
        del arg41_1
        del buf20
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf22, arg42_1, arg43_1, arg44_1, arg45_1, 953344, grid=grid(953344), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        buf23 = empty_strided_cuda((8, 38, 56, 56), (119168, 1, 2128, 38), torch.float32)
        # Topologically Sorted Source Nodes: [x_183], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf22, buf15, buf23, 953344, grid=grid(953344), stream=stream0)
        del buf22
        # Topologically Sorted Source Nodes: [x_183, x_184], Original ATen: [aten.cat, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 228, 56, 56), (715008, 1, 12768, 228))
        del arg46_1
        del buf23
        buf25 = buf24; del buf24  # reuse
        buf26 = empty_strided_cuda((8, 228, 56, 56), (715008, 1, 12768, 228), torch.float32)
        # Topologically Sorted Source Nodes: [x_185, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf25, arg47_1, arg48_1, arg49_1, arg50_1, buf26, 5720064, grid=grid(5720064), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        del buf25
        # Topologically Sorted Source Nodes: [x_186, x_187], Original ATen: [aten.silu, aten.convolution]
        buf27 = extern_kernels.convolution(buf26, arg51_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=228, bias=None)
        assert_size_stride(buf27, (8, 228, 28, 28), (178752, 1, 6384, 228))
        del arg51_1
        del buf26
        buf28 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf28, arg52_1, arg53_1, arg54_1, arg55_1, 1430016, grid=grid(1430016), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        buf29 = empty_strided_cuda((8, 228, 1, 1, 7), (1596, 1, 12768, 12768, 228), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_52], Original ATen: [aten.mean]
        triton_red_fused_mean_14.run(buf28, buf29, 12768, 112, grid=grid(12768), stream=stream0)
        buf31 = empty_strided_cuda((8, 228, 1, 1), (228, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_52], Original ATen: [aten.mean]
        triton_per_fused_mean_15.run(buf29, buf31, 1824, 7, grid=grid(1824), stream=stream0)
        del buf29
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53], Original ATen: [aten.mean, aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 19, 1, 1), (19, 1, 1, 1))
        del arg56_1
        del buf31
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53, batch_norm_73, x_se_54], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_16.run(buf33, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, 152, grid=grid(152), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del arg61_1
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53, batch_norm_73, x_se_54, x_se_55], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf34 = extern_kernels.convolution(buf33, arg62_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 228, 1, 1), (228, 1, 1, 1))
        del arg62_1
        del buf33
        buf35 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53, batch_norm_73, x_se_54, x_se_55, sigmoid_13, x_189, x_190], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_17.run(buf35, buf34, arg63_1, 1430016, grid=grid(1430016), stream=stream0)
        del arg63_1
        del buf34
        # Topologically Sorted Source Nodes: [x_se_52, x_se_53, batch_norm_73, x_se_54, x_se_55, sigmoid_13, x_189, x_190, x_191], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf36 = extern_kernels.convolution(buf35, arg64_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 50, 28, 28), (39200, 1, 1400, 50))
        del arg64_1
        del buf35
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf37, arg65_1, arg66_1, arg67_1, arg68_1, 313600, grid=grid(313600), stream=stream0)
        del arg65_1
        del arg66_1
        del arg67_1
        del arg68_1
        # Topologically Sorted Source Nodes: [x_193], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 300, 28, 28), (235200, 1, 8400, 300))
        del arg69_1
        buf39 = buf38; del buf38  # reuse
        buf40 = empty_strided_cuda((8, 300, 28, 28), (235200, 1, 8400, 300), torch.float32)
        # Topologically Sorted Source Nodes: [x_194, x_195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_19.run(buf39, arg70_1, arg71_1, arg72_1, arg73_1, buf40, 1881600, grid=grid(1881600), stream=stream0)
        del arg70_1
        del arg71_1
        del arg72_1
        del arg73_1
        del buf39
        # Topologically Sorted Source Nodes: [x_195, x_196], Original ATen: [aten.silu, aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg74_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=300, bias=None)
        assert_size_stride(buf41, (8, 300, 28, 28), (235200, 1, 8400, 300))
        del arg74_1
        del buf40
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_20.run(buf42, arg75_1, arg76_1, arg77_1, arg78_1, 1881600, grid=grid(1881600), stream=stream0)
        del arg75_1
        del arg76_1
        del arg77_1
        del arg78_1
        buf43 = empty_strided_cuda((8, 300, 1, 1, 7), (2100, 1, 16800, 16800, 300), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_56], Original ATen: [aten.mean]
        triton_red_fused_mean_21.run(buf42, buf43, 16800, 112, grid=grid(16800), stream=stream0)
        buf45 = empty_strided_cuda((8, 300, 1, 1), (300, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_56], Original ATen: [aten.mean]
        triton_per_fused_mean_22.run(buf43, buf45, 2400, 7, grid=grid(2400), stream=stream0)
        del buf43
        # Topologically Sorted Source Nodes: [x_se_56, x_se_57], Original ATen: [aten.mean, aten.convolution]
        buf46 = extern_kernels.convolution(buf45, arg79_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 25, 1, 1), (25, 1, 1, 1))
        del arg79_1
        del buf45
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_se_56, x_se_57, batch_norm_77, x_se_58], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_23.run(buf47, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, 200, grid=grid(200), stream=stream0)
        del arg80_1
        del arg81_1
        del arg82_1
        del arg83_1
        del arg84_1
        # Topologically Sorted Source Nodes: [x_se_56, x_se_57, batch_norm_77, x_se_58, x_se_59], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf48 = extern_kernels.convolution(buf47, arg85_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 300, 1, 1), (300, 1, 1, 1))
        del arg85_1
        del buf47
        buf49 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_se_56, x_se_57, batch_norm_77, x_se_58, x_se_59, sigmoid_14, x_198, x_199], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_24.run(buf49, buf48, arg86_1, 1881600, grid=grid(1881600), stream=stream0)
        del arg86_1
        del buf48
        # Topologically Sorted Source Nodes: [x_se_56, x_se_57, batch_norm_77, x_se_58, x_se_59, sigmoid_14, x_198, x_199, x_200], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf50 = extern_kernels.convolution(buf49, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 61, 28, 28), (47824, 1, 1708, 61))
        del arg87_1
        del buf49
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_201], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf51, arg88_1, arg89_1, arg90_1, arg91_1, 382592, grid=grid(382592), stream=stream0)
        del arg88_1
        del arg89_1
        del arg90_1
        del arg91_1
        buf52 = empty_strided_cuda((8, 61, 28, 28), (47824, 1, 1708, 61), torch.float32)
        # Topologically Sorted Source Nodes: [x_202], Original ATen: [aten.cat]
        triton_poi_fused_cat_26.run(buf51, buf37, buf52, 382592, grid=grid(382592), stream=stream0)
        del buf37
        del buf51
        # Topologically Sorted Source Nodes: [x_202, x_203], Original ATen: [aten.cat, aten.convolution]
        buf53 = extern_kernels.convolution(buf52, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (8, 366, 28, 28), (286944, 1, 10248, 366))
        del arg92_1
        del buf52
        buf54 = buf53; del buf53  # reuse
        buf55 = empty_strided_cuda((8, 366, 28, 28), (286944, 1, 10248, 366), torch.float32)
        # Topologically Sorted Source Nodes: [x_204, x_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_27.run(buf54, arg93_1, arg94_1, arg95_1, arg96_1, buf55, 2295552, grid=grid(2295552), stream=stream0)
        del arg93_1
        del arg94_1
        del arg95_1
        del arg96_1
        del buf54
        # Topologically Sorted Source Nodes: [x_205, x_206], Original ATen: [aten.silu, aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg97_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=366, bias=None)
        assert_size_stride(buf56, (8, 366, 14, 14), (71736, 1, 5124, 366))
        del arg97_1
        del buf55
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_207], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf57, arg98_1, arg99_1, arg100_1, arg101_1, 573888, grid=grid(573888), stream=stream0)
        del arg100_1
        del arg101_1
        del arg98_1
        del arg99_1
        buf58 = empty_strided_cuda((8, 366, 1, 1, 2), (732, 1, 5856, 5856, 366), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_60], Original ATen: [aten.mean]
        triton_red_fused_mean_29.run(buf57, buf58, 5856, 98, grid=grid(5856), stream=stream0)
        buf60 = empty_strided_cuda((8, 366, 1, 1), (366, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_60], Original ATen: [aten.mean]
        triton_per_fused_mean_30.run(buf58, buf60, 2928, 2, grid=grid(2928), stream=stream0)
        del buf58
        # Topologically Sorted Source Nodes: [x_se_60, x_se_61], Original ATen: [aten.mean, aten.convolution]
        buf61 = extern_kernels.convolution(buf60, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (8, 30, 1, 1), (30, 1, 1, 1))
        del arg102_1
        del buf60
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_se_60, x_se_61, batch_norm_81, x_se_62], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_31.run(buf62, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, 240, grid=grid(240), stream=stream0)
        del arg103_1
        del arg104_1
        del arg105_1
        del arg106_1
        del arg107_1
        # Topologically Sorted Source Nodes: [x_se_60, x_se_61, batch_norm_81, x_se_62, x_se_63], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf63 = extern_kernels.convolution(buf62, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 366, 1, 1), (366, 1, 1, 1))
        del arg108_1
        del buf62
        buf64 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_se_60, x_se_61, batch_norm_81, x_se_62, x_se_63, sigmoid_15, x_208, x_209], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_32.run(buf64, buf63, arg109_1, 573888, grid=grid(573888), stream=stream0)
        del arg109_1
        del buf63
        # Topologically Sorted Source Nodes: [x_se_60, x_se_61, batch_norm_81, x_se_62, x_se_63, sigmoid_15, x_208, x_209, x_210], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf65 = extern_kernels.convolution(buf64, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 72, 14, 14), (14112, 1, 1008, 72))
        del arg110_1
        del buf64
        buf66 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf66, arg111_1, arg112_1, arg113_1, arg114_1, 112896, grid=grid(112896), stream=stream0)
        del arg111_1
        del arg112_1
        del arg113_1
        del arg114_1
        # Topologically Sorted Source Nodes: [x_212], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 432, 14, 14), (84672, 1, 6048, 432))
        del arg115_1
        buf68 = buf67; del buf67  # reuse
        buf69 = reinterpret_tensor(buf15, (8, 432, 14, 14), (84672, 1, 6048, 432), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_213, x_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf68, arg116_1, arg117_1, arg118_1, arg119_1, buf69, 677376, grid=grid(677376), stream=stream0)
        del arg116_1
        del arg117_1
        del arg118_1
        del arg119_1
        del buf68
        # Topologically Sorted Source Nodes: [x_214, x_215], Original ATen: [aten.silu, aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg120_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf70, (8, 432, 14, 14), (84672, 1, 6048, 432))
        del arg120_1
        del buf69
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_216], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_35.run(buf71, arg121_1, arg122_1, arg123_1, arg124_1, 677376, grid=grid(677376), stream=stream0)
        del arg121_1
        del arg122_1
        del arg123_1
        del arg124_1
        buf72 = empty_strided_cuda((8, 432, 1, 1, 2), (864, 1, 6912, 6912, 432), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_64], Original ATen: [aten.mean]
        triton_red_fused_mean_36.run(buf71, buf72, 6912, 98, grid=grid(6912), stream=stream0)
        buf74 = empty_strided_cuda((8, 432, 1, 1), (432, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_64], Original ATen: [aten.mean]
        triton_per_fused_mean_37.run(buf72, buf74, 3456, 2, grid=grid(3456), stream=stream0)
        del buf72
        # Topologically Sorted Source Nodes: [x_se_64, x_se_65], Original ATen: [aten.mean, aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 36, 1, 1), (36, 1, 1, 1))
        del arg125_1
        del buf74
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_se_64, x_se_65, batch_norm_85, x_se_66], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_38.run(buf76, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, 288, grid=grid(288), stream=stream0)
        del arg126_1
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        # Topologically Sorted Source Nodes: [x_se_64, x_se_65, batch_norm_85, x_se_66, x_se_67], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf77 = extern_kernels.convolution(buf76, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 432, 1, 1), (432, 1, 1, 1))
        del arg131_1
        del buf76
        buf78 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_se_64, x_se_65, batch_norm_85, x_se_66, x_se_67, sigmoid_16, x_217, x_218], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_39.run(buf78, buf77, arg132_1, 677376, grid=grid(677376), stream=stream0)
        del arg132_1
        del buf77
        # Topologically Sorted Source Nodes: [x_se_64, x_se_65, batch_norm_85, x_se_66, x_se_67, sigmoid_16, x_217, x_218, x_219], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf79 = extern_kernels.convolution(buf78, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 84, 14, 14), (16464, 1, 1176, 84))
        del arg133_1
        del buf78
        buf80 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [x_220], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_40.run(buf80, arg134_1, arg135_1, arg136_1, arg137_1, 131712, grid=grid(131712), stream=stream0)
        del arg134_1
        del arg135_1
        del arg136_1
        del arg137_1
        buf81 = empty_strided_cuda((8, 84, 14, 14), (16464, 1, 1176, 84), torch.float32)
        # Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.cat]
        triton_poi_fused_cat_41.run(buf80, buf66, buf81, 131712, grid=grid(131712), stream=stream0)
        del buf66
        del buf80
        # Topologically Sorted Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 504, 14, 14), (98784, 1, 7056, 504))
        del arg138_1
        buf83 = buf82; del buf82  # reuse
        buf84 = empty_strided_cuda((8, 504, 14, 14), (98784, 1, 7056, 504), torch.float32)
        # Topologically Sorted Source Nodes: [x_223, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_42.run(buf83, arg139_1, arg140_1, arg141_1, arg142_1, buf84, 790272, grid=grid(790272), stream=stream0)
        del arg139_1
        del arg140_1
        del arg141_1
        del arg142_1
        del buf83
        # Topologically Sorted Source Nodes: [x_224, x_225], Original ATen: [aten.silu, aten.convolution]
        buf85 = extern_kernels.convolution(buf84, arg143_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=504, bias=None)
        assert_size_stride(buf85, (8, 504, 14, 14), (98784, 1, 7056, 504))
        del arg143_1
        del buf84
        buf86 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_226], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf86, arg144_1, arg145_1, arg146_1, arg147_1, 790272, grid=grid(790272), stream=stream0)
        del arg144_1
        del arg145_1
        del arg146_1
        del arg147_1
        buf87 = empty_strided_cuda((8, 504, 1, 1, 2), (1008, 1, 8064, 8064, 504), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_68], Original ATen: [aten.mean]
        triton_red_fused_mean_44.run(buf86, buf87, 8064, 98, grid=grid(8064), stream=stream0)
        buf89 = empty_strided_cuda((8, 504, 1, 1), (504, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_68], Original ATen: [aten.mean]
        triton_per_fused_mean_45.run(buf87, buf89, 4032, 2, grid=grid(4032), stream=stream0)
        del buf87
        # Topologically Sorted Source Nodes: [x_se_68, x_se_69], Original ATen: [aten.mean, aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 42, 1, 1), (42, 1, 1, 1))
        del arg148_1
        del buf89
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_se_68, x_se_69, batch_norm_89, x_se_70], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_46.run(buf91, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, 336, grid=grid(336), stream=stream0)
        del arg149_1
        del arg150_1
        del arg151_1
        del arg152_1
        del arg153_1
        # Topologically Sorted Source Nodes: [x_se_68, x_se_69, batch_norm_89, x_se_70, x_se_71], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf92 = extern_kernels.convolution(buf91, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 504, 1, 1), (504, 1, 1, 1))
        del arg154_1
        del buf91
        buf93 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_se_68, x_se_69, batch_norm_89, x_se_70, x_se_71, sigmoid_17, x_227, x_228], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_47.run(buf93, buf92, arg155_1, 790272, grid=grid(790272), stream=stream0)
        del arg155_1
        del buf92
        # Topologically Sorted Source Nodes: [x_se_68, x_se_69, batch_norm_89, x_se_70, x_se_71, sigmoid_17, x_227, x_228, x_229], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf94 = extern_kernels.convolution(buf93, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 95, 14, 14), (18620, 1, 1330, 95))
        del arg156_1
        del buf93
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_48.run(buf95, arg157_1, arg158_1, arg159_1, arg160_1, 148960, grid=grid(148960), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        buf96 = empty_strided_cuda((8, 95, 14, 14), (18620, 1, 1330, 95), torch.float32)
        # Topologically Sorted Source Nodes: [x_231], Original ATen: [aten.cat]
        triton_poi_fused_cat_49.run(buf95, buf81, buf96, 148960, grid=grid(148960), stream=stream0)
        del buf81
        del buf95
        # Topologically Sorted Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 570, 14, 14), (111720, 1, 7980, 570))
        del arg161_1
        buf98 = buf97; del buf97  # reuse
        buf99 = empty_strided_cuda((8, 570, 14, 14), (111720, 1, 7980, 570), torch.float32)
        # Topologically Sorted Source Nodes: [x_233, x_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_50.run(buf98, arg162_1, arg163_1, arg164_1, arg165_1, buf99, 893760, grid=grid(893760), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        del buf98
        # Topologically Sorted Source Nodes: [x_234, x_235], Original ATen: [aten.silu, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg166_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=570, bias=None)
        assert_size_stride(buf100, (8, 570, 14, 14), (111720, 1, 7980, 570))
        del arg166_1
        del buf99
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_51.run(buf101, arg167_1, arg168_1, arg169_1, arg170_1, 893760, grid=grid(893760), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        buf102 = empty_strided_cuda((8, 570, 1, 1, 2), (1140, 1, 9120, 9120, 570), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_72], Original ATen: [aten.mean]
        triton_red_fused_mean_52.run(buf101, buf102, 9120, 98, grid=grid(9120), stream=stream0)
        buf104 = empty_strided_cuda((8, 570, 1, 1), (570, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_72], Original ATen: [aten.mean]
        triton_per_fused_mean_53.run(buf102, buf104, 4560, 2, grid=grid(4560), stream=stream0)
        del buf102
        # Topologically Sorted Source Nodes: [x_se_72, x_se_73], Original ATen: [aten.mean, aten.convolution]
        buf105 = extern_kernels.convolution(buf104, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (8, 47, 1, 1), (47, 1, 1, 1))
        del arg171_1
        del buf104
        buf106 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_se_72, x_se_73, batch_norm_93, x_se_74], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_54.run(buf106, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, 376, grid=grid(376), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        del arg176_1
        # Topologically Sorted Source Nodes: [x_se_72, x_se_73, batch_norm_93, x_se_74, x_se_75], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf107 = extern_kernels.convolution(buf106, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 570, 1, 1), (570, 1, 1, 1))
        del arg177_1
        del buf106
        buf108 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_se_72, x_se_73, batch_norm_93, x_se_74, x_se_75, sigmoid_18, x_237, x_238], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_55.run(buf108, buf107, arg178_1, 893760, grid=grid(893760), stream=stream0)
        del arg178_1
        del buf107
        # Topologically Sorted Source Nodes: [x_se_72, x_se_73, batch_norm_93, x_se_74, x_se_75, sigmoid_18, x_237, x_238, x_239], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf109 = extern_kernels.convolution(buf108, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 106, 14, 14), (20776, 1, 1484, 106))
        del arg179_1
        del buf108
        buf110 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_240], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_56.run(buf110, arg180_1, arg181_1, arg182_1, arg183_1, 166208, grid=grid(166208), stream=stream0)
        del arg180_1
        del arg181_1
        del arg182_1
        del arg183_1
        buf111 = empty_strided_cuda((8, 106, 14, 14), (20776, 1, 1484, 106), torch.float32)
        # Topologically Sorted Source Nodes: [x_241], Original ATen: [aten.cat]
        triton_poi_fused_cat_57.run(buf110, buf96, buf111, 166208, grid=grid(166208), stream=stream0)
        del buf110
        del buf96
        # Topologically Sorted Source Nodes: [x_242], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 636, 14, 14), (124656, 1, 8904, 636))
        del arg184_1
        buf113 = buf112; del buf112  # reuse
        buf114 = empty_strided_cuda((8, 636, 14, 14), (124656, 1, 8904, 636), torch.float32)
        # Topologically Sorted Source Nodes: [x_243, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_58.run(buf113, arg185_1, arg186_1, arg187_1, arg188_1, buf114, 997248, grid=grid(997248), stream=stream0)
        del arg185_1
        del arg186_1
        del arg187_1
        del arg188_1
        del buf113
        # Topologically Sorted Source Nodes: [x_244, x_245], Original ATen: [aten.silu, aten.convolution]
        buf115 = extern_kernels.convolution(buf114, arg189_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=636, bias=None)
        assert_size_stride(buf115, (8, 636, 14, 14), (124656, 1, 8904, 636))
        del arg189_1
        del buf114
        buf116 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_246], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_59.run(buf116, arg190_1, arg191_1, arg192_1, arg193_1, 997248, grid=grid(997248), stream=stream0)
        del arg190_1
        del arg191_1
        del arg192_1
        del arg193_1
        buf117 = empty_strided_cuda((8, 636, 1, 1, 2), (1272, 1, 10176, 10176, 636), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_76], Original ATen: [aten.mean]
        triton_red_fused_mean_60.run(buf116, buf117, 10176, 98, grid=grid(10176), stream=stream0)
        buf119 = empty_strided_cuda((8, 636, 1, 1), (636, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_76], Original ATen: [aten.mean]
        triton_per_fused_mean_61.run(buf117, buf119, 5088, 2, grid=grid(5088), stream=stream0)
        del buf117
        # Topologically Sorted Source Nodes: [x_se_76, x_se_77], Original ATen: [aten.mean, aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 53, 1, 1), (53, 1, 1, 1))
        del arg194_1
        del buf119
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [x_se_76, x_se_77, batch_norm_97, x_se_78], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_62.run(buf121, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, 424, grid=grid(424), stream=stream0)
        del arg195_1
        del arg196_1
        del arg197_1
        del arg198_1
        del arg199_1
        # Topologically Sorted Source Nodes: [x_se_76, x_se_77, batch_norm_97, x_se_78, x_se_79], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf122 = extern_kernels.convolution(buf121, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 636, 1, 1), (636, 1, 1, 1))
        del arg200_1
        del buf121
        buf123 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_se_76, x_se_77, batch_norm_97, x_se_78, x_se_79, sigmoid_19, x_247, x_248], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_63.run(buf123, buf122, arg201_1, 997248, grid=grid(997248), stream=stream0)
        del arg201_1
        del buf122
        # Topologically Sorted Source Nodes: [x_se_76, x_se_77, batch_norm_97, x_se_78, x_se_79, sigmoid_19, x_247, x_248, x_249], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf124 = extern_kernels.convolution(buf123, arg202_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 117, 14, 14), (22932, 1, 1638, 117))
        del arg202_1
        del buf123
        buf125 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_250], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_64.run(buf125, arg203_1, arg204_1, arg205_1, arg206_1, 183456, grid=grid(183456), stream=stream0)
        del arg203_1
        del arg204_1
        del arg205_1
        del arg206_1
        buf126 = empty_strided_cuda((8, 117, 14, 14), (22932, 1, 1638, 117), torch.float32)
        # Topologically Sorted Source Nodes: [x_251], Original ATen: [aten.cat]
        triton_poi_fused_cat_65.run(buf125, buf111, buf126, 183456, grid=grid(183456), stream=stream0)
        del buf111
        del buf125
        # Topologically Sorted Source Nodes: [x_252], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 702, 14, 14), (137592, 1, 9828, 702))
        del arg207_1
        buf128 = buf127; del buf127  # reuse
        buf129 = empty_strided_cuda((8, 702, 14, 14), (137592, 1, 9828, 702), torch.float32)
        # Topologically Sorted Source Nodes: [x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_66.run(buf128, arg208_1, arg209_1, arg210_1, arg211_1, buf129, 1100736, grid=grid(1100736), stream=stream0)
        del arg208_1
        del arg209_1
        del arg210_1
        del arg211_1
        del buf128
        # Topologically Sorted Source Nodes: [x_254, x_255], Original ATen: [aten.silu, aten.convolution]
        buf130 = extern_kernels.convolution(buf129, arg212_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=702, bias=None)
        assert_size_stride(buf130, (8, 702, 14, 14), (137592, 1, 9828, 702))
        del arg212_1
        del buf129
        buf131 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_256], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_67.run(buf131, arg213_1, arg214_1, arg215_1, arg216_1, 1100736, grid=grid(1100736), stream=stream0)
        del arg213_1
        del arg214_1
        del arg215_1
        del arg216_1
        buf132 = empty_strided_cuda((8, 702, 1, 1, 2), (1404, 1, 11232, 11232, 702), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_80], Original ATen: [aten.mean]
        triton_red_fused_mean_68.run(buf131, buf132, 11232, 98, grid=grid(11232), stream=stream0)
        buf134 = empty_strided_cuda((8, 702, 1, 1), (702, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_80], Original ATen: [aten.mean]
        triton_per_fused_mean_69.run(buf132, buf134, 5616, 2, grid=grid(5616), stream=stream0)
        del buf132
        # Topologically Sorted Source Nodes: [x_se_80, x_se_81], Original ATen: [aten.mean, aten.convolution]
        buf135 = extern_kernels.convolution(buf134, arg217_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (8, 58, 1, 1), (58, 1, 1, 1))
        del arg217_1
        del buf134
        buf136 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [x_se_80, x_se_81, batch_norm_101, x_se_82], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_70.run(buf136, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, 464, grid=grid(464), stream=stream0)
        del arg218_1
        del arg219_1
        del arg220_1
        del arg221_1
        del arg222_1
        # Topologically Sorted Source Nodes: [x_se_80, x_se_81, batch_norm_101, x_se_82, x_se_83], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf137 = extern_kernels.convolution(buf136, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 702, 1, 1), (702, 1, 1, 1))
        del arg223_1
        del buf136
        buf138 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_se_80, x_se_81, batch_norm_101, x_se_82, x_se_83, sigmoid_20, x_257, x_258], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_71.run(buf138, buf137, arg224_1, 1100736, grid=grid(1100736), stream=stream0)
        del arg224_1
        del buf137
        # Topologically Sorted Source Nodes: [x_se_80, x_se_81, batch_norm_101, x_se_82, x_se_83, sigmoid_20, x_257, x_258, x_259], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf139 = extern_kernels.convolution(buf138, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del arg225_1
        del buf138
        buf140 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_260], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_72.run(buf140, arg226_1, arg227_1, arg228_1, arg229_1, 200704, grid=grid(200704), stream=stream0)
        del arg226_1
        del arg227_1
        del arg228_1
        del arg229_1
        buf141 = empty_strided_cuda((8, 128, 14, 14), (25088, 1, 1792, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_261], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf140, buf126, buf141, 200704, grid=grid(200704), stream=stream0)
        del buf126
        del buf140
        # Topologically Sorted Source Nodes: [x_261, x_262], Original ATen: [aten.cat, aten.convolution]
        buf142 = extern_kernels.convolution(buf141, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 768, 14, 14), (150528, 1, 10752, 768))
        del arg230_1
        del buf141
        buf143 = buf142; del buf142  # reuse
        buf144 = reinterpret_tensor(buf0, (8, 768, 14, 14), (150528, 1, 10752, 768), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_263, x_264], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_74.run(buf143, arg231_1, arg232_1, arg233_1, arg234_1, buf144, 1204224, grid=grid(1204224), stream=stream0)
        del arg231_1
        del arg232_1
        del arg233_1
        del arg234_1
        del buf143
        # Topologically Sorted Source Nodes: [x_264, x_265], Original ATen: [aten.silu, aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg235_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf145, (8, 768, 7, 7), (37632, 1, 5376, 768))
        del arg235_1
        del buf144
        buf146 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [x_266], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_75.run(buf146, arg236_1, arg237_1, arg238_1, arg239_1, 301056, grid=grid(301056), stream=stream0)
        del arg236_1
        del arg237_1
        del arg238_1
        del arg239_1
        buf148 = empty_strided_cuda((8, 768, 1, 1), (768, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_84], Original ATen: [aten.mean]
        triton_per_fused_mean_76.run(buf146, buf148, 6144, 49, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_84, x_se_85], Original ATen: [aten.mean, aten.convolution]
        buf149 = extern_kernels.convolution(buf148, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg240_1
        del buf148
        buf150 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [x_se_84, x_se_85, batch_norm_105, x_se_86], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_77.run(buf150, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, 512, grid=grid(512), stream=stream0)
        del arg241_1
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        # Topologically Sorted Source Nodes: [x_se_84, x_se_85, batch_norm_105, x_se_86, x_se_87], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf151 = extern_kernels.convolution(buf150, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg246_1
        del buf150
        buf152 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_se_84, x_se_85, batch_norm_105, x_se_86, x_se_87, sigmoid_21, x_267, x_268], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_78.run(buf152, buf151, arg247_1, 301056, grid=grid(301056), stream=stream0)
        del arg247_1
        del buf151
        # Topologically Sorted Source Nodes: [x_se_84, x_se_85, batch_norm_105, x_se_86, x_se_87, sigmoid_21, x_267, x_268, x_269], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf153 = extern_kernels.convolution(buf152, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 140, 7, 7), (6860, 1, 980, 140))
        del arg248_1
        del buf152
        buf154 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_79.run(buf154, arg249_1, arg250_1, arg251_1, arg252_1, 54880, grid=grid(54880), stream=stream0)
        del arg249_1
        del arg250_1
        del arg251_1
        del arg252_1
        # Topologically Sorted Source Nodes: [x_271], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, arg253_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (8, 840, 7, 7), (41160, 1, 5880, 840))
        del arg253_1
        buf156 = buf155; del buf155  # reuse
        buf157 = empty_strided_cuda((8, 840, 7, 7), (41160, 1, 5880, 840), torch.float32)
        # Topologically Sorted Source Nodes: [x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_80.run(buf156, arg254_1, arg255_1, arg256_1, arg257_1, buf157, 329280, grid=grid(329280), stream=stream0)
        del arg254_1
        del arg255_1
        del arg256_1
        del arg257_1
        del buf156
        # Topologically Sorted Source Nodes: [x_273, x_274], Original ATen: [aten.silu, aten.convolution]
        buf158 = extern_kernels.convolution(buf157, arg258_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=840, bias=None)
        assert_size_stride(buf158, (8, 840, 7, 7), (41160, 1, 5880, 840))
        del arg258_1
        del buf157
        buf159 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [x_275], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_81.run(buf159, arg259_1, arg260_1, arg261_1, arg262_1, 329280, grid=grid(329280), stream=stream0)
        del arg259_1
        del arg260_1
        del arg261_1
        del arg262_1
        buf161 = empty_strided_cuda((8, 840, 1, 1), (840, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_88], Original ATen: [aten.mean]
        triton_per_fused_mean_82.run(buf159, buf161, 6720, 49, grid=grid(6720), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_88, x_se_89], Original ATen: [aten.mean, aten.convolution]
        buf162 = extern_kernels.convolution(buf161, arg263_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (8, 70, 1, 1), (70, 1, 1, 1))
        del arg263_1
        del buf161
        buf163 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_se_88, x_se_89, batch_norm_109, x_se_90], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_83.run(buf163, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, 560, grid=grid(560), stream=stream0)
        del arg264_1
        del arg265_1
        del arg266_1
        del arg267_1
        del arg268_1
        # Topologically Sorted Source Nodes: [x_se_88, x_se_89, batch_norm_109, x_se_90, x_se_91], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf164 = extern_kernels.convolution(buf163, arg269_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 840, 1, 1), (840, 1, 1, 1))
        del arg269_1
        del buf163
        buf165 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_se_88, x_se_89, batch_norm_109, x_se_90, x_se_91, sigmoid_22, x_276, x_277], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_84.run(buf165, buf164, arg270_1, 329280, grid=grid(329280), stream=stream0)
        del arg270_1
        del buf164
        # Topologically Sorted Source Nodes: [x_se_88, x_se_89, batch_norm_109, x_se_90, x_se_91, sigmoid_22, x_276, x_277, x_278], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf166 = extern_kernels.convolution(buf165, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 151, 7, 7), (7399, 1, 1057, 151))
        del arg271_1
        del buf165
        buf167 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_85.run(buf167, arg272_1, arg273_1, arg274_1, arg275_1, 59192, grid=grid(59192), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        buf168 = empty_strided_cuda((8, 151, 7, 7), (7399, 1, 1057, 151), torch.float32)
        # Topologically Sorted Source Nodes: [x_280], Original ATen: [aten.cat]
        triton_poi_fused_cat_86.run(buf167, buf154, buf168, 59192, grid=grid(59192), stream=stream0)
        del buf154
        del buf167
        # Topologically Sorted Source Nodes: [x_281], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 906, 7, 7), (44394, 1, 6342, 906))
        del arg276_1
        buf170 = buf169; del buf169  # reuse
        buf171 = empty_strided_cuda((8, 906, 7, 7), (44394, 1, 6342, 906), torch.float32)
        # Topologically Sorted Source Nodes: [x_282, x_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_87.run(buf170, arg277_1, arg278_1, arg279_1, arg280_1, buf171, 355152, grid=grid(355152), stream=stream0)
        del arg277_1
        del arg278_1
        del arg279_1
        del arg280_1
        del buf170
        # Topologically Sorted Source Nodes: [x_283, x_284], Original ATen: [aten.silu, aten.convolution]
        buf172 = extern_kernels.convolution(buf171, arg281_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=906, bias=None)
        assert_size_stride(buf172, (8, 906, 7, 7), (44394, 1, 6342, 906))
        del arg281_1
        del buf171
        buf173 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_88.run(buf173, arg282_1, arg283_1, arg284_1, arg285_1, 355152, grid=grid(355152), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        buf175 = empty_strided_cuda((8, 906, 1, 1), (906, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_92], Original ATen: [aten.mean]
        triton_per_fused_mean_89.run(buf173, buf175, 7248, 49, grid=grid(7248), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_92, x_se_93], Original ATen: [aten.mean, aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 75, 1, 1), (75, 1, 1, 1))
        del arg286_1
        del buf175
        buf177 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [x_se_92, x_se_93, batch_norm_113, x_se_94], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_90.run(buf177, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, 600, grid=grid(600), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        del arg291_1
        # Topologically Sorted Source Nodes: [x_se_92, x_se_93, batch_norm_113, x_se_94, x_se_95], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf178 = extern_kernels.convolution(buf177, arg292_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 906, 1, 1), (906, 1, 1, 1))
        del arg292_1
        del buf177
        buf179 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [x_se_92, x_se_93, batch_norm_113, x_se_94, x_se_95, sigmoid_23, x_286, x_287], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_91.run(buf179, buf178, arg293_1, 355152, grid=grid(355152), stream=stream0)
        del arg293_1
        del buf178
        # Topologically Sorted Source Nodes: [x_se_92, x_se_93, batch_norm_113, x_se_94, x_se_95, sigmoid_23, x_286, x_287, x_288], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf180 = extern_kernels.convolution(buf179, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 162, 7, 7), (7938, 1, 1134, 162))
        del arg294_1
        del buf179
        buf181 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [x_289], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_92.run(buf181, arg295_1, arg296_1, arg297_1, arg298_1, 63504, grid=grid(63504), stream=stream0)
        del arg295_1
        del arg296_1
        del arg297_1
        del arg298_1
        buf182 = empty_strided_cuda((8, 162, 7, 7), (7938, 1, 1134, 162), torch.float32)
        # Topologically Sorted Source Nodes: [x_290], Original ATen: [aten.cat]
        triton_poi_fused_cat_93.run(buf181, buf168, buf182, 63504, grid=grid(63504), stream=stream0)
        del buf168
        del buf181
        # Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, arg299_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 972, 7, 7), (47628, 1, 6804, 972))
        del arg299_1
        buf184 = buf183; del buf183  # reuse
        buf185 = empty_strided_cuda((8, 972, 7, 7), (47628, 1, 6804, 972), torch.float32)
        # Topologically Sorted Source Nodes: [x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_94.run(buf184, arg300_1, arg301_1, arg302_1, arg303_1, buf185, 381024, grid=grid(381024), stream=stream0)
        del arg300_1
        del arg301_1
        del arg302_1
        del arg303_1
        del buf184
        # Topologically Sorted Source Nodes: [x_293, x_294], Original ATen: [aten.silu, aten.convolution]
        buf186 = extern_kernels.convolution(buf185, arg304_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=972, bias=None)
        assert_size_stride(buf186, (8, 972, 7, 7), (47628, 1, 6804, 972))
        del arg304_1
        del buf185
        buf187 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_295], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_95.run(buf187, arg305_1, arg306_1, arg307_1, arg308_1, 381024, grid=grid(381024), stream=stream0)
        del arg305_1
        del arg306_1
        del arg307_1
        del arg308_1
        buf189 = empty_strided_cuda((8, 972, 1, 1), (972, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_96], Original ATen: [aten.mean]
        triton_per_fused_mean_96.run(buf187, buf189, 7776, 49, grid=grid(7776), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_96, x_se_97], Original ATen: [aten.mean, aten.convolution]
        buf190 = extern_kernels.convolution(buf189, arg309_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (8, 81, 1, 1), (81, 1, 1, 1))
        del arg309_1
        del buf189
        buf191 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_se_96, x_se_97, batch_norm_117, x_se_98], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_97.run(buf191, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, 648, grid=grid(648), stream=stream0)
        del arg310_1
        del arg311_1
        del arg312_1
        del arg313_1
        del arg314_1
        # Topologically Sorted Source Nodes: [x_se_96, x_se_97, batch_norm_117, x_se_98, x_se_99], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf192 = extern_kernels.convolution(buf191, arg315_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 972, 1, 1), (972, 1, 1, 1))
        del arg315_1
        del buf191
        buf193 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [x_se_96, x_se_97, batch_norm_117, x_se_98, x_se_99, sigmoid_24, x_296, x_297], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_98.run(buf193, buf192, arg316_1, 381024, grid=grid(381024), stream=stream0)
        del arg316_1
        del buf192
        # Topologically Sorted Source Nodes: [x_se_96, x_se_97, batch_norm_117, x_se_98, x_se_99, sigmoid_24, x_296, x_297, x_298], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf194 = extern_kernels.convolution(buf193, arg317_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 174, 7, 7), (8526, 1, 1218, 174))
        del arg317_1
        del buf193
        buf195 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [x_299], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_99.run(buf195, arg318_1, arg319_1, arg320_1, arg321_1, 68208, grid=grid(68208), stream=stream0)
        del arg318_1
        del arg319_1
        del arg320_1
        del arg321_1
        buf196 = empty_strided_cuda((8, 174, 7, 7), (8526, 1, 1218, 174), torch.float32)
        # Topologically Sorted Source Nodes: [x_300], Original ATen: [aten.cat]
        triton_poi_fused_cat_100.run(buf195, buf182, buf196, 68208, grid=grid(68208), stream=stream0)
        del buf182
        del buf195
        # Topologically Sorted Source Nodes: [x_301], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, arg322_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
        del arg322_1
        buf198 = buf197; del buf197  # reuse
        buf199 = empty_strided_cuda((8, 1044, 7, 7), (51156, 1, 7308, 1044), torch.float32)
        # Topologically Sorted Source Nodes: [x_302, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_101.run(buf198, arg323_1, arg324_1, arg325_1, arg326_1, buf199, 409248, grid=grid(409248), stream=stream0)
        del arg323_1
        del arg324_1
        del arg325_1
        del arg326_1
        del buf198
        # Topologically Sorted Source Nodes: [x_303, x_304], Original ATen: [aten.silu, aten.convolution]
        buf200 = extern_kernels.convolution(buf199, arg327_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1044, bias=None)
        assert_size_stride(buf200, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
        del arg327_1
        del buf199
        buf201 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [x_305], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_102.run(buf201, arg328_1, arg329_1, arg330_1, arg331_1, 409248, grid=grid(409248), stream=stream0)
        del arg328_1
        del arg329_1
        del arg330_1
        del arg331_1
        buf203 = empty_strided_cuda((8, 1044, 1, 1), (1044, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_100], Original ATen: [aten.mean]
        triton_per_fused_mean_103.run(buf201, buf203, 8352, 49, grid=grid(8352), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_100, x_se_101], Original ATen: [aten.mean, aten.convolution]
        buf204 = extern_kernels.convolution(buf203, arg332_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 87, 1, 1), (87, 1, 1, 1))
        del arg332_1
        del buf203
        buf205 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_se_100, x_se_101, batch_norm_121, x_se_102], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_104.run(buf205, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, 696, grid=grid(696), stream=stream0)
        del arg333_1
        del arg334_1
        del arg335_1
        del arg336_1
        del arg337_1
        # Topologically Sorted Source Nodes: [x_se_100, x_se_101, batch_norm_121, x_se_102, x_se_103], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf206 = extern_kernels.convolution(buf205, arg338_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 1044, 1, 1), (1044, 1, 1, 1))
        del arg338_1
        del buf205
        buf207 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [x_se_100, x_se_101, batch_norm_121, x_se_102, x_se_103, sigmoid_25, x_306, x_307], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_mean_mul_relu_sigmoid_105.run(buf207, buf206, arg339_1, 409248, grid=grid(409248), stream=stream0)
        del arg339_1
        del buf206
        # Topologically Sorted Source Nodes: [x_se_100, x_se_101, batch_norm_121, x_se_102, x_se_103, sigmoid_25, x_306, x_307, x_308], Original ATen: [aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu, aten.sigmoid, aten.mul, aten.hardtanh]
        buf208 = extern_kernels.convolution(buf207, arg340_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 185, 7, 7), (9065, 1, 1295, 185))
        del arg340_1
        del buf207
        buf209 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [x_309], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_106.run(buf209, arg341_1, arg342_1, arg343_1, arg344_1, 72520, grid=grid(72520), stream=stream0)
        del arg341_1
        del arg342_1
        del arg343_1
        del arg344_1
        buf210 = empty_strided_cuda((8, 185, 7, 7), (9065, 1, 1295, 185), torch.float32)
        # Topologically Sorted Source Nodes: [x_310], Original ATen: [aten.cat]
        triton_poi_fused_cat_107.run(buf209, buf196, buf210, 72520, grid=grid(72520), stream=stream0)
        del buf196
        del buf209
        # Topologically Sorted Source Nodes: [x_310, x_311], Original ATen: [aten.cat, aten.convolution]
        buf211 = extern_kernels.convolution(buf210, arg345_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
        del arg345_1
        del buf210
        buf212 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [x_312], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_108.run(buf212, arg346_1, arg347_1, arg348_1, arg349_1, 501760, grid=grid(501760), stream=stream0)
        del arg346_1
        del arg347_1
        del arg348_1
        del arg349_1
        buf214 = empty_strided_cuda((8, 1280, 1, 1), (1280, 1, 10240, 10240), torch.float32)
        # Topologically Sorted Source Nodes: [x_313, x_314], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_109.run(buf212, buf214, 10240, 49, grid=grid(10240), stream=stream0)
        del buf212
        buf215 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_317], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg351_1, reinterpret_tensor(buf214, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg350_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf215)
        del arg350_1
        del arg351_1
        del buf214
    return (buf215, )


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
    arg26_1 = rand_strided((27, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((162, 27, 1, 1), (27, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((162, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((38, 162, 1, 1), (162, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((228, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((228, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((19, 228, 1, 1), (228, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((228, 19, 1, 1), (19, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((50, 228, 1, 1), (228, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((300, 50, 1, 1), (50, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((300, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((25, 300, 1, 1), (300, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((300, 25, 1, 1), (25, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((61, 300, 1, 1), (300, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((366, 61, 1, 1), (61, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((366, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((30, 366, 1, 1), (366, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((366, 30, 1, 1), (30, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((72, 366, 1, 1), (366, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((432, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((36, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((432, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((84, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((504, 84, 1, 1), (84, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((504, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((42, 504, 1, 1), (504, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((504, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((95, 504, 1, 1), (504, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((570, 95, 1, 1), (95, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((570, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((47, 570, 1, 1), (570, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((570, 47, 1, 1), (47, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((106, 570, 1, 1), (570, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((636, 106, 1, 1), (106, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((636, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((53, 636, 1, 1), (636, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((636, 53, 1, 1), (53, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((117, 636, 1, 1), (636, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((702, 117, 1, 1), (117, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((702, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((58, 702, 1, 1), (702, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((702, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((128, 702, 1, 1), (702, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((768, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((64, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((768, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((140, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((840, 140, 1, 1), (140, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((70, 840, 1, 1), (840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((840, 70, 1, 1), (70, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((151, 840, 1, 1), (840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((906, 151, 1, 1), (151, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((906, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((75, 906, 1, 1), (906, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((906, 75, 1, 1), (75, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((162, 906, 1, 1), (906, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((972, 162, 1, 1), (162, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((972, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((81, 972, 1, 1), (972, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((972, 81, 1, 1), (81, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((174, 972, 1, 1), (972, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1044, 174, 1, 1), (174, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1044, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((87, 1044, 1, 1), (1044, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1044, 87, 1, 1), (87, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((185, 1044, 1, 1), (1044, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1280, 185, 1, 1), (185, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('rexnet_100', benchmark_compiled_module)
