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
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_9 => convolution_52
# Graph fragment:
#   %convolution_52 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/jp/cjpg4fkzjl3bxzg2py5zajjht2of6nfbla3smj2khcxixgipugzk.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_9 => convolution_52
# Graph fragment:
#   %convolution_52 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
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


# kernel path: /tmp/torchinductor_sahanp/y2/cy2ah26jhsl2ezwj3kdl6rmo7gwhj3ahmvv3odqbsjtdv2uve4oc.py
# Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.gelu]
# Source node to ATen node mapping:
#   input_10 => add_366, mul_495, mul_496, sub_128
#   input_11 => add_367, erf_52, mul_497, mul_498, mul_499
# Graph fragment:
#   %sub_128 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_226), kwargs = {})
#   %mul_495 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_128, %unsqueeze_228), kwargs = {})
#   %mul_496 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_495, %unsqueeze_230), kwargs = {})
#   %add_366 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_496, %unsqueeze_232), kwargs = {})
#   %mul_497 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_366, 0.5), kwargs = {})
#   %mul_498 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_366, 0.7071067811865476), kwargs = {})
#   %erf_52 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_498,), kwargs = {})
#   %add_367 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_52, 1), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_497, %add_367), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_gelu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_gelu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_gelu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_gelu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
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
    tmp16 = 0.5
    tmp17 = tmp15 * tmp16
    tmp18 = 0.7071067811865476
    tmp19 = tmp15 * tmp18
    tmp20 = libdevice.erf(tmp19)
    tmp21 = tmp20 + tmp9
    tmp22 = tmp17 * tmp21
    tl.store(out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cs/ccs2pvhyyweifpqjauw7oead5hhhqc3l2fhgihhixedjtqacedc4.py
# Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.gelu, aten.convolution]
# Source node to ATen node mapping:
#   input_11 => add_367, erf_52, mul_497, mul_498, mul_499
#   input_12 => convolution_53
# Graph fragment:
#   %mul_497 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_366, 0.5), kwargs = {})
#   %mul_498 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_366, 0.7071067811865476), kwargs = {})
#   %erf_52 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_498,), kwargs = {})
#   %add_367 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_52, 1), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_497, %add_367), kwargs = {})
#   %convolution_53 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_499, %arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_gelu_3 = async_compile.triton('triton_poi_fused_convolution_gelu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/fb/cfbsdtsl5bmv5mqsubbc3cjarnhydlqzsqfgschwk5cdjan7dozp.py
# Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.gelu]
# Source node to ATen node mapping:
#   input_13 => add_369, mul_501, mul_502, sub_129
#   input_14 => add_370, erf_53, mul_503, mul_504, mul_505
# Graph fragment:
#   %sub_129 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_234), kwargs = {})
#   %mul_501 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %unsqueeze_236), kwargs = {})
#   %mul_502 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_501, %unsqueeze_238), kwargs = {})
#   %add_369 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_502, %unsqueeze_240), kwargs = {})
#   %mul_503 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_369, 0.5), kwargs = {})
#   %mul_504 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_369, 0.7071067811865476), kwargs = {})
#   %erf_53 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_504,), kwargs = {})
#   %add_370 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_53, 1), kwargs = {})
#   %mul_505 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_503, %add_370), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_gelu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_gelu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_gelu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9633792
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
    tmp16 = 0.5
    tmp17 = tmp15 * tmp16
    tmp18 = 0.7071067811865476
    tmp19 = tmp15 * tmp18
    tmp20 = libdevice.erf(tmp19)
    tmp21 = tmp20 + tmp9
    tmp22 = tmp17 * tmp21
    tl.store(out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ah/cahtnitd5l7glgdkt6vejqhsf7nkz2rg6yif7xumux3fhvpxoavu.py
# Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten.gelu, aten.convolution]
# Source node to ATen node mapping:
#   input_14 => add_370, erf_53, mul_503, mul_504, mul_505
#   input_15 => convolution_54
# Graph fragment:
#   %mul_503 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_369, 0.5), kwargs = {})
#   %mul_504 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_369, 0.7071067811865476), kwargs = {})
#   %erf_53 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_504,), kwargs = {})
#   %add_370 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_53, 1), kwargs = {})
#   %mul_505 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_503, %add_370), kwargs = {})
#   %convolution_54 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_505, %arg11_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_gelu_5 = async_compile.triton('triton_poi_fused_convolution_gelu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_gelu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 294912
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


# kernel path: /tmp/torchinductor_sahanp/52/c52qd7ldbxr46js2nhe4yedbycgkr2ecsqheir3qdgzuxgnm74be.py
# Topologically Sorted Source Nodes: [stack_3], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   stack_3 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_254, %unsqueeze_255], 4), kwargs = {})
triton_poi_fused_stack_6 = async_compile.triton('triton_poi_fused_stack_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x3 = (xindex // 896)
    x1 = (xindex // 2) % 16
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + x3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 28.000001
    tmp8 = tmp6 / tmp7
    tmp9 = 6.283185307179586
    tmp10 = tmp8 * tmp9
    tmp11 = 2*x1
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 0.5
    tmp14 = tmp12 * tmp13
    tmp15 = libdevice.floor(tmp14)
    tmp16 = 2.0
    tmp17 = tmp15 * tmp16
    tmp18 = 0.03125
    tmp19 = tmp17 * tmp18
    tmp20 = 10000.0
    tmp21 = libdevice.pow(tmp20, tmp19)
    tmp22 = tmp10 / tmp21
    tmp23 = tl_math.sin(tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp4, tmp23, tmp24)
    tmp26 = tmp0 >= tmp3
    tmp27 = tl.full([1], 2, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = 1 + (2*x1)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp13
    tmp32 = libdevice.floor(tmp31)
    tmp33 = tmp32 * tmp16
    tmp34 = tmp33 * tmp18
    tmp35 = libdevice.pow(tmp20, tmp34)
    tmp36 = tmp10 / tmp35
    tmp37 = tl_math.cos(tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp26, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp25, tmp39)
    tl.store(out_ptr0 + (x6), tmp40, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ia/ciasa4xjdowie3c2nbvsskmylj4ev3rjkisdvhnkcjm546mtufwj.py
# Topologically Sorted Source Nodes: [stack_2], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   stack_2 => cat_8
# Graph fragment:
#   %cat_8 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_252, %unsqueeze_253], 4), kwargs = {})
triton_poi_fused_stack_7 = async_compile.triton('triton_poi_fused_stack_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x2 = (xindex // 32) % 28
    x1 = (xindex // 2) % 16
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + x2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 28.000001
    tmp8 = tmp6 / tmp7
    tmp9 = 6.283185307179586
    tmp10 = tmp8 * tmp9
    tmp11 = 2*x1
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 0.5
    tmp14 = tmp12 * tmp13
    tmp15 = libdevice.floor(tmp14)
    tmp16 = 2.0
    tmp17 = tmp15 * tmp16
    tmp18 = 0.03125
    tmp19 = tmp17 * tmp18
    tmp20 = 10000.0
    tmp21 = libdevice.pow(tmp20, tmp19)
    tmp22 = tmp10 / tmp21
    tmp23 = tl_math.sin(tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp4, tmp23, tmp24)
    tmp26 = tmp0 >= tmp3
    tmp27 = tl.full([1], 2, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = 1 + (2*x1)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp13
    tmp32 = libdevice.floor(tmp31)
    tmp33 = tmp32 * tmp16
    tmp34 = tmp33 * tmp18
    tmp35 = libdevice.pow(tmp20, tmp34)
    tmp36 = tmp10 / tmp35
    tmp37 = tl_math.cos(tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp26, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp25, tmp39)
    tl.store(out_ptr0 + (x6), tmp40, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/af/caf4wdpqsepa4viloidmb4a7gxxra5ii5maa3d7xkqo7rce6g4m3.py
# Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_6 => cat_10
# Graph fragment:
#   %cat_10 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_466, %view_465], 3), kwargs = {})
triton_poi_fused_cat_8 = async_compile.triton('triton_poi_fused_cat_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((32*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 64, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((32*x1) + ((-32) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/df/cdfm6bzmxqda37linrdtlwhsuzpseu4wm4tm3snhl6tkscl7r3p6.py
# Topologically Sorted Source Nodes: [x_459, layer_norm_77], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_77 => add_376, add_377, clone_273, mul_512, mul_513, rsqrt_77, sub_131, var_mean_77
#   x_459 => add_375
# Graph fragment:
#   %add_375 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_240, %permute_242), kwargs = {})
#   %clone_273 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_375,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_77 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_273, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_131 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_273, %getitem_235), kwargs = {})
#   %add_376 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_234, 1e-06), kwargs = {})
#   %rsqrt_77 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_376,), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_131, %rsqrt_77), kwargs = {})
#   %mul_513 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_512, %arg19_1), kwargs = {})
#   %add_377 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_513, %arg20_1), kwargs = {})
triton_red_fused_add_native_layer_norm_9 = async_compile.triton('triton_red_fused_add_native_layer_norm_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 784
    tmp21_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (784*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp18 = tmp16 + tmp17
        tmp19 = tmp15 + tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp21_mean_next, tmp21_m2_next, tmp21_weight_next = triton_helpers.welford_reduce(
            tmp20, tmp21_mean, tmp21_m2, tmp21_weight, roffset == 0
        )
        tmp21_mean = tl.where(rmask & xmask, tmp21_mean_next, tmp21_mean)
        tmp21_m2 = tl.where(rmask & xmask, tmp21_m2_next, tmp21_m2)
        tmp21_weight = tl.where(rmask & xmask, tmp21_weight_next, tmp21_weight)
        tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp19, rmask & xmask)
    tmp21_tmp, tmp22_tmp, tmp23_tmp = triton_helpers.welford(
        tmp21_mean, tmp21_m2, tmp21_weight, 1
    )
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp24 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tmp24 - tmp21
        tmp26 = 768.0
        tmp27 = tmp22 / tmp26
        tmp28 = 1e-06
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp33 = tmp31 * tmp32
        tmp35 = tmp33 + tmp34
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gx/cgxdsuanmzoquexe2sxjxyuuiepjmja62vg2567ad6ils6hxuxtf.py
# Topologically Sorted Source Nodes: [q_51], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   q_51 => pow_99, sum_73
# Graph fragment:
#   %pow_99 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%getitem_236, 2.0), kwargs = {})
#   %sum_73 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_99, [-1], True), kwargs = {})
triton_red_fused_linalg_vector_norm_10 = async_compile.triton('triton_red_fused_linalg_vector_norm_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_linalg_vector_norm_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_linalg_vector_norm_10(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43008
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2304*r2) + (258048*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/do/cdoosqr6s45prwqs7dqw67sr5vez3irsqs5ero3r677ahnfz4wg4.py
# Topologically Sorted Source Nodes: [q_51], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   q_51 => pow_99, sum_73
# Graph fragment:
#   %pow_99 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%getitem_236, 2.0), kwargs = {})
#   %sum_73 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_99, [-1], True), kwargs = {})
triton_per_fused_linalg_vector_norm_11 = async_compile.triton('triton_per_fused_linalg_vector_norm_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_11(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 7
    RBLOCK: tl.constexpr = 8
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (5376*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rw/crwi3k7kxskw2ubiulnmomzobibo3jxxsnydfrqudyks44663pv4.py
# Topologically Sorted Source Nodes: [k_51], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   k_51 => pow_101, sum_74
# Graph fragment:
#   %pow_101 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%getitem_237, 2.0), kwargs = {})
#   %sum_74 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_101, [-1], True), kwargs = {})
triton_red_fused_linalg_vector_norm_12 = async_compile.triton('triton_red_fused_linalg_vector_norm_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_linalg_vector_norm_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_linalg_vector_norm_12(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43008
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp1 = tl.load(in_ptr1 + (768 + x0), xmask, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (768 + x0 + (2304*r2) + (258048*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uw/cuwmhzheay7sidyuvf5z4x3plorp4zsttvfobppqsjbelkbarj4m.py
# Topologically Sorted Source Nodes: [q_51, matmul_48], Original ATen: [aten.div, aten.clone]
# Source node to ATen node mapping:
#   matmul_48 => clone_274
#   q_51 => div_84
# Graph fragment:
#   %div_84 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%getitem_236, %expand_145), kwargs = {})
#   %clone_274 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_147,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_div_13 = async_compile.triton('triton_poi_fused_clone_div_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_div_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_div_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (2304*x2) + (1806336*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = libdevice.sqrt(tmp3)
    tmp5 = 1e-12
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp2 / tmp6
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bp/cbpkrspimwrgs3fnqlrdyzkz7tp7pl6t3fgx4bhwrxffr3erfgiq.py
# Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_48 => clone_275
# Graph fragment:
#   %clone_275 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_148,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_14 = async_compile.triton('triton_poi_fused_clone_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_14(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 48
    x1 = (xindex // 48) % 784
    x2 = (xindex // 37632) % 16
    x3 = (xindex // 602112)
    x4 = (xindex // 37632)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (48*x2) + (2304*x1) + (1806336*x3)), None)
    tmp1 = tl.load(in_ptr1 + (768 + x0 + (48*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (48*x4)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = libdevice.sqrt(tmp3)
    tmp5 = 1e-12
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp2 / tmp6
    tl.store(out_ptr0 + (x5), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sp/cspx2f5qwu4mgewjm3tfcapvwsw54xtdeihlbdg4otmteb5suwfv.py
# Topologically Sorted Source Nodes: [attn_73], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_73 => div_86, exp_24, sum_75
# Graph fragment:
#   %ge_scalar_23 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%arg23_1, 0), kwargs = {})
#   %scalar_tensor_default_23 : [num_users=2] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (1,), kwargs = {dtype: torch.float32, device: cuda:0, pin_memory: False})
#   %neg_default_23 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%scalar_tensor_default_23,), kwargs = {})
#   %where_self_23 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%ge_scalar_23, %scalar_tensor_default_23, %neg_default_23), kwargs = {})
#   %mul_tensor_69 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_473, %where_self_23), kwargs = {})
#   %amax_default_23 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_69, [-1], True), kwargs = {})
#   %sub_tensor_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_69, %amax_default_23), kwargs = {})
#   %mul_tensor_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self_23, %arg23_1), kwargs = {})
#   %mul_tensor_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_23, %mul_tensor_70), kwargs = {})
#   %exp_24 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_71,), kwargs = {})
#   %sum_75 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [-1], True), kwargs = {})
#   %div_86 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_24, %sum_75), kwargs = {})
triton_per_fused__softmax_15 = async_compile.triton('triton_per_fused__softmax_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_15(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 48
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x1 = (xindex // 48) % 16
    tmp0 = tl.load(in_ptr0 + (r3 + (48*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 >= tmp2
    tmp4 = 1.0
    tmp5 = -1.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, float("-inf"))
    tmp11 = triton_helpers.max2(tmp10, 1)[:, None]
    tmp12 = tmp7 - tmp11
    tmp13 = tmp6 * tmp1
    tmp14 = tmp12 * tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tmp15 / tmp19
    tl.store(out_ptr2 + (r3 + (48*x4)), tmp20, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sl/cslnwsfc2w77xauqum6rv4eib4ud4o3fwudyhqdxu4ip32hwmels.py
# Topologically Sorted Source Nodes: [x_461], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_461 => clone_277
# Graph fragment:
#   %clone_277 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_150,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_16 = async_compile.triton('triton_poi_fused_clone_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_16(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (1536 + y0 + (2304*x2) + (1806336*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1536 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vn/cvnt4folxp3fbomzthzkmopjtv5notg2yaijg4dciixmcv5mqog7.py
# Topologically Sorted Source Nodes: [x_463], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_463 => clone_278
# Graph fragment:
#   %clone_278 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_477,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_17 = async_compile.triton('triton_poi_fused_clone_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/k2/ck2gpwo6dcwx26rv4z3ukrhqvhblawjwigwvu7h46hkv3egejkn3.py
# Topologically Sorted Source Nodes: [x_463, mul_107, x_465, layer_norm_78], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_78 => add_380, add_381, clone_280, mul_516, mul_517, rsqrt_78, sub_133, var_mean_78
#   mul_107 => mul_515
#   x_463 => add_378
#   x_465 => add_379
# Graph fragment:
#   %add_378 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_479, %arg25_1), kwargs = {})
#   %mul_515 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg18_1, %add_378), kwargs = {})
#   %add_379 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_375, %mul_515), kwargs = {})
#   %clone_280 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_379,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_78 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_280, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_133 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_280, %getitem_240), kwargs = {})
#   %add_380 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_239, 1e-06), kwargs = {})
#   %rsqrt_78 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_380,), kwargs = {})
#   %mul_516 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_133, %rsqrt_78), kwargs = {})
#   %mul_517 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_516, %arg27_1), kwargs = {})
#   %add_381 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_517, %arg28_1), kwargs = {})
triton_per_fused_add_mul_native_layer_norm_18 = async_compile.triton('triton_per_fused_add_mul_native_layer_norm_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mul_native_layer_norm_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp33, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xk/cxkhgydcelu5gexrfo27lkndt5ajqxzfg6mw2xxqzvrwi3ffhhcn.py
# Topologically Sorted Source Nodes: [x_467, x_468, x_469], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_467 => convolution_56
#   x_468 => add_382, erf_54, mul_518, mul_519, mul_520
#   x_469 => add_384, mul_522, mul_523, sub_134
# Graph fragment:
#   %convolution_56 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%view_480, %arg29_1, %arg30_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 768), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_56, 0.5), kwargs = {})
#   %mul_519 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_56, 0.7071067811865476), kwargs = {})
#   %erf_54 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_519,), kwargs = {})
#   %add_382 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_54, 1), kwargs = {})
#   %mul_520 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_518, %add_382), kwargs = {})
#   %sub_134 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_520, %unsqueeze_257), kwargs = {})
#   %mul_522 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_134, %unsqueeze_259), kwargs = {})
#   %mul_523 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_522, %unsqueeze_261), kwargs = {})
#   %add_384 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_523, %unsqueeze_263), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = tmp18 * tmp8
    tmp20 = tmp12 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mw/cmw5c4a7xobhm4bcraplsy7yly6cb2v3kjmjmv4nzhnbjtagdu7g.py
# Topologically Sorted Source Nodes: [x_463, mul_107, x_465, mul_108, x_472, layer_norm_79], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_79 => add_386, add_387, clone_281, mul_525, mul_526, rsqrt_79, sub_135, var_mean_79
#   mul_107 => mul_515
#   mul_108 => mul_524
#   x_463 => add_378
#   x_465 => add_379
#   x_472 => add_385
# Graph fragment:
#   %add_378 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_479, %arg25_1), kwargs = {})
#   %mul_515 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg18_1, %add_378), kwargs = {})
#   %add_379 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_375, %mul_515), kwargs = {})
#   %mul_524 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg26_1, %permute_249), kwargs = {})
#   %add_385 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_379, %mul_524), kwargs = {})
#   %clone_281 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_385,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_79 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_281, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_135 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_281, %getitem_242), kwargs = {})
#   %add_386 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_241, 1e-06), kwargs = {})
#   %rsqrt_79 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_386,), kwargs = {})
#   %mul_525 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_135, %rsqrt_79), kwargs = {})
#   %mul_526 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_525, %arg38_1), kwargs = {})
#   %add_387 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_526, %arg39_1), kwargs = {})
triton_per_fused_add_mul_native_layer_norm_20 = async_compile.triton('triton_per_fused_add_mul_native_layer_norm_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 9, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mul_native_layer_norm_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp12, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp39, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/c7/cc7xnzl42wfpnm4zp47kh77nf4fcvwiivomq5ni3s5ym3hl7siiq.py
# Topologically Sorted Source Nodes: [x_474], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_474 => add_388, erf_55, mul_527, mul_528, mul_529
# Graph fragment:
#   %mul_527 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_483, 0.5), kwargs = {})
#   %mul_528 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_483, 0.7071067811865476), kwargs = {})
#   %erf_55 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_528,), kwargs = {})
#   %add_388 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_55, 1), kwargs = {})
#   %mul_529 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_527, %add_388), kwargs = {})
triton_poi_fused_gelu_21 = async_compile.triton('triton_poi_fused_gelu_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ki/ckifkyrr73sjpv7kluvfu7ijjdmz66lkizj3orna6hyjaibjxtco.py
# Topologically Sorted Source Nodes: [x_893, x_norm1_2], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_893 => cat_11
#   x_norm1_2 => add_712, add_713, mul_968, mul_969, rsqrt_149, sub_251, var_mean_149
# Graph fragment:
#   %cat_11 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_289, %add_711], 1), kwargs = {})
#   %var_mean_149 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_251 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_11, %getitem_451), kwargs = {})
#   %add_712 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_450, 1e-06), kwargs = {})
#   %rsqrt_149 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_712,), kwargs = {})
#   %mul_968 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_251, %rsqrt_149), kwargs = {})
#   %mul_969 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_968, %arg643_1), kwargs = {})
#   %add_713 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_969, %arg644_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_22 = async_compile.triton('triton_per_fused_cat_native_layer_norm_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 10, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 785
    r2 = rindex
    x1 = (xindex // 785)
    x3 = xindex
    tmp48 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr9 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 785, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (768*((-1) + x0)) + (602112*x1)), rmask & tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr3 + (r2 + (768*(((-1) + x0) % 784)) + (602112*x1)), rmask & tmp6, other=0.0)
    tmp12 = tl.load(in_ptr4 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 * tmp13
    tmp15 = tmp9 + tmp14
    tmp16 = tl.load(in_ptr5 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr6 + (r2 + (768*((-1) + x0)) + (602112*x1)), rmask & tmp6, other=0.0)
    tmp18 = tl.load(in_ptr7 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tmp15 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp6, tmp21, tmp22)
    tmp24 = tl.where(tmp4, tmp5, tmp23)
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tl.full([1], 768, tl.int32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 / tmp33
    tmp35 = tmp25 - tmp34
    tmp36 = tmp35 * tmp35
    tmp37 = tl.broadcast_to(tmp36, [RBLOCK])
    tmp39 = tl.where(rmask, tmp37, 0)
    tmp40 = triton_helpers.promote_to_tensor(tl.sum(tmp39, 0))
    tmp41 = tmp24 - tmp34
    tmp42 = 768.0
    tmp43 = tmp40 / tmp42
    tmp44 = 1e-06
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.rsqrt(tmp45)
    tmp47 = tmp41 * tmp46
    tmp49 = tmp47 * tmp48
    tmp51 = tmp49 + tmp50
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp24, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp51, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2x/c2xsygsxncvdxpbf7anngaty3dgt3utdurtzi5yfzuyc7qdupr44.py
# Topologically Sorted Source Nodes: [x_attn_2, mul_202, x_894, x_895], Original ATen: [aten.cat, aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   mul_202 => mul_970
#   x_894 => add_714
#   x_895 => add_715, add_716, mul_971, mul_972, rsqrt_150, sub_252, var_mean_150
#   x_attn_2 => cat_12
# Graph fragment:
#   %cat_12 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_909, %slice_74], 1), kwargs = {})
#   %mul_970 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg653_1, %cat_12), kwargs = {})
#   %add_714 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_11, %mul_970), kwargs = {})
#   %var_mean_150 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_714, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_252 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_714, %getitem_457), kwargs = {})
#   %add_715 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_456, 1e-06), kwargs = {})
#   %rsqrt_150 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_715,), kwargs = {})
#   %mul_971 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_252, %rsqrt_150), kwargs = {})
#   %mul_972 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_971, %arg654_1), kwargs = {})
#   %add_716 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_972, %arg655_1), kwargs = {})
triton_per_fused_add_cat_mul_native_layer_norm_23 = async_compile.triton('triton_per_fused_add_cat_mul_native_layer_norm_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_mul_native_layer_norm_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_mul_native_layer_norm_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 785
    x1 = (xindex // 785)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = x0
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 >= tmp3
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp2 < tmp5
    tmp7 = tl.load(in_ptr2 + (r2 + (768*x1)), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp2 >= tmp5
    tmp9 = tl.full([1], 785, tl.int64)
    tmp10 = tmp2 < tmp9
    tmp11 = tl.load(in_ptr3 + (768 + r2 + (768*((-1) + x0)) + (602880*x1)), rmask & tmp8, other=0.0)
    tmp12 = tl.where(tmp6, tmp7, tmp11)
    tmp13 = tmp1 * tmp12
    tmp14 = tmp0 + tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tl.full([1], 768, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp15 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = tmp14 - tmp24
    tmp32 = 768.0
    tmp33 = tmp30 / tmp32
    tmp34 = 1e-06
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp41, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dk/cdkt22phq5jjcxe35a42zem32bd6icjaz2dh6l4oudublkmgid5z.py
# Topologically Sorted Source Nodes: [x_896, x_897], Original ATen: [aten.add, aten.gelu]
# Source node to ATen node mapping:
#   x_896 => add_717
#   x_897 => add_718, erf_102, mul_973, mul_974, mul_975
# Graph fragment:
#   %add_717 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_911, %arg658_1), kwargs = {})
#   %mul_973 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_717, 0.5), kwargs = {})
#   %mul_974 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_717, 0.7071067811865476), kwargs = {})
#   %erf_102 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_974,), kwargs = {})
#   %add_718 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_102, 1), kwargs = {})
#   %mul_975 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_973, %add_718), kwargs = {})
triton_poi_fused_add_gelu_24 = async_compile.triton('triton_poi_fused_add_gelu_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_gelu_24(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3o/c3o2udvk3q5brcikjkzoau6b4bbvx742ggaeytfcyhulwgjurr2g.py
# Topologically Sorted Source Nodes: [x_901, x_902, x_norm1_3], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_901 => cat_13
#   x_902 => add_719
#   x_norm1_3 => add_720, add_721, mul_977, mul_978, rsqrt_151, sub_253, var_mean_151
# Graph fragment:
#   %cat_13 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_976, %slice_78], 1), kwargs = {})
#   %add_719 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_716, %cat_13), kwargs = {})
#   %var_mean_151 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_719, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_253 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_719, %getitem_459), kwargs = {})
#   %add_720 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_458, 1e-06), kwargs = {})
#   %rsqrt_151 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_720,), kwargs = {})
#   %mul_977 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_253, %rsqrt_151), kwargs = {})
#   %mul_978 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_977, %arg661_1), kwargs = {})
#   %add_721 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_978, %arg662_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_25 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 785
    x1 = (xindex // 785)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask, other=0.0)
    tmp42 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp5, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r2 + (768*x1)), rmask & tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp5, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tmp1 >= tmp4
    tmp14 = tl.full([1], 785, tl.int64)
    tmp15 = tmp1 < tmp14
    tmp16 = tl.load(in_ptr0 + (768 + r2 + (768*((-1) + x0)) + (602880*x1)), rmask & tmp13, other=0.0)
    tmp17 = tl.where(tmp5, tmp12, tmp16)
    tmp18 = tmp0 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 768, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tmp18 - tmp28
    tmp36 = 768.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp18, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp45, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vs/cvsoi2jflzzngkyrchk466gewogke23t6yivc4qpipdkimbdh644.py
# Topologically Sorted Source Nodes: [x_910, x_911, x_912], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_910 => cat_15
#   x_911 => add_727
#   x_912 => var_mean_153
# Graph fragment:
#   %cat_15 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_985, %slice_85], 1), kwargs = {})
#   %add_727 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_724, %cat_15), kwargs = {})
#   %var_mean_153 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_727, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_cat_native_layer_norm_26 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 785
    x1 = (xindex // 785)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask, other=0.0)
    tmp1 = x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp5, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r2 + (768*x1)), rmask & tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp5, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tmp1 >= tmp4
    tmp14 = tl.full([1], 785, tl.int64)
    tmp15 = tmp1 < tmp14
    tmp16 = tl.load(in_ptr0 + (768 + r2 + (768*((-1) + x0)) + (602880*x1)), rmask & tmp13, other=0.0)
    tmp17 = tl.where(tmp5, tmp12, tmp16)
    tmp18 = tmp0 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 768, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp18, rmask)
    tl.store(out_ptr1 + (x3), tmp28, None)
    tl.store(out_ptr2 + (x3), tmp34, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fo/cfoturmitry7i36tmll3g6ikvvfxdeww2y7fku2uqxrnwjijjrma.py
# Topologically Sorted Source Nodes: [x_914], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_914 => clone_543
# Graph fragment:
#   %clone_543 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%select_5,), kwargs = {})
triton_poi_fused_clone_27 = async_compile.triton('triton_poi_fused_clone_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (602880*x1)), xmask)
    tmp1 = tl.load(in_ptr1 + (785*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (785*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (192, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg2_1, (192, ), (1, ))
    assert_size_stride(arg3_1, (192, ), (1, ))
    assert_size_stride(arg4_1, (192, ), (1, ))
    assert_size_stride(arg5_1, (192, ), (1, ))
    assert_size_stride(arg6_1, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg7_1, (384, ), (1, ))
    assert_size_stride(arg8_1, (384, ), (1, ))
    assert_size_stride(arg9_1, (384, ), (1, ))
    assert_size_stride(arg10_1, (384, ), (1, ))
    assert_size_stride(arg11_1, (768, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (2304, 768), (768, 1))
    assert_size_stride(arg22_1, (2304, ), (1, ))
    assert_size_stride(arg23_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg24_1, (768, 768), (768, 1))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (3072, 768), (768, 1))
    assert_size_stride(arg41_1, (3072, ), (1, ))
    assert_size_stride(arg42_1, (768, 3072), (3072, 1))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (2304, 768), (768, 1))
    assert_size_stride(arg48_1, (2304, ), (1, ))
    assert_size_stride(arg49_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg50_1, (768, 768), (768, 1))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (3072, 768), (768, 1))
    assert_size_stride(arg67_1, (3072, ), (1, ))
    assert_size_stride(arg68_1, (768, 3072), (3072, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (2304, 768), (768, 1))
    assert_size_stride(arg74_1, (2304, ), (1, ))
    assert_size_stride(arg75_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg76_1, (768, 768), (768, 1))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (3072, 768), (768, 1))
    assert_size_stride(arg93_1, (3072, ), (1, ))
    assert_size_stride(arg94_1, (768, 3072), (3072, 1))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (2304, 768), (768, 1))
    assert_size_stride(arg100_1, (2304, ), (1, ))
    assert_size_stride(arg101_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg102_1, (768, 768), (768, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (3072, 768), (768, 1))
    assert_size_stride(arg119_1, (3072, ), (1, ))
    assert_size_stride(arg120_1, (768, 3072), (3072, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (2304, 768), (768, 1))
    assert_size_stride(arg126_1, (2304, ), (1, ))
    assert_size_stride(arg127_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg128_1, (768, 768), (768, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (3072, 768), (768, 1))
    assert_size_stride(arg145_1, (3072, ), (1, ))
    assert_size_stride(arg146_1, (768, 3072), (3072, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (2304, 768), (768, 1))
    assert_size_stride(arg152_1, (2304, ), (1, ))
    assert_size_stride(arg153_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg154_1, (768, 768), (768, 1))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (3072, 768), (768, 1))
    assert_size_stride(arg171_1, (3072, ), (1, ))
    assert_size_stride(arg172_1, (768, 3072), (3072, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (768, ), (1, ))
    assert_size_stride(arg177_1, (2304, 768), (768, 1))
    assert_size_stride(arg178_1, (2304, ), (1, ))
    assert_size_stride(arg179_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg180_1, (768, 768), (768, 1))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg192_1, (768, ), (1, ))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (3072, 768), (768, 1))
    assert_size_stride(arg197_1, (3072, ), (1, ))
    assert_size_stride(arg198_1, (768, 3072), (3072, 1))
    assert_size_stride(arg199_1, (768, ), (1, ))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (768, ), (1, ))
    assert_size_stride(arg203_1, (2304, 768), (768, 1))
    assert_size_stride(arg204_1, (2304, ), (1, ))
    assert_size_stride(arg205_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg206_1, (768, 768), (768, 1))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (768, ), (1, ))
    assert_size_stride(arg209_1, (768, ), (1, ))
    assert_size_stride(arg210_1, (768, ), (1, ))
    assert_size_stride(arg211_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg212_1, (768, ), (1, ))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (768, ), (1, ))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (768, ), (1, ))
    assert_size_stride(arg217_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg218_1, (768, ), (1, ))
    assert_size_stride(arg219_1, (768, ), (1, ))
    assert_size_stride(arg220_1, (768, ), (1, ))
    assert_size_stride(arg221_1, (768, ), (1, ))
    assert_size_stride(arg222_1, (3072, 768), (768, 1))
    assert_size_stride(arg223_1, (3072, ), (1, ))
    assert_size_stride(arg224_1, (768, 3072), (3072, 1))
    assert_size_stride(arg225_1, (768, ), (1, ))
    assert_size_stride(arg226_1, (768, ), (1, ))
    assert_size_stride(arg227_1, (768, ), (1, ))
    assert_size_stride(arg228_1, (768, ), (1, ))
    assert_size_stride(arg229_1, (2304, 768), (768, 1))
    assert_size_stride(arg230_1, (2304, ), (1, ))
    assert_size_stride(arg231_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg232_1, (768, 768), (768, 1))
    assert_size_stride(arg233_1, (768, ), (1, ))
    assert_size_stride(arg234_1, (768, ), (1, ))
    assert_size_stride(arg235_1, (768, ), (1, ))
    assert_size_stride(arg236_1, (768, ), (1, ))
    assert_size_stride(arg237_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg238_1, (768, ), (1, ))
    assert_size_stride(arg239_1, (768, ), (1, ))
    assert_size_stride(arg240_1, (768, ), (1, ))
    assert_size_stride(arg241_1, (768, ), (1, ))
    assert_size_stride(arg242_1, (768, ), (1, ))
    assert_size_stride(arg243_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg244_1, (768, ), (1, ))
    assert_size_stride(arg245_1, (768, ), (1, ))
    assert_size_stride(arg246_1, (768, ), (1, ))
    assert_size_stride(arg247_1, (768, ), (1, ))
    assert_size_stride(arg248_1, (3072, 768), (768, 1))
    assert_size_stride(arg249_1, (3072, ), (1, ))
    assert_size_stride(arg250_1, (768, 3072), (3072, 1))
    assert_size_stride(arg251_1, (768, ), (1, ))
    assert_size_stride(arg252_1, (768, ), (1, ))
    assert_size_stride(arg253_1, (768, ), (1, ))
    assert_size_stride(arg254_1, (768, ), (1, ))
    assert_size_stride(arg255_1, (2304, 768), (768, 1))
    assert_size_stride(arg256_1, (2304, ), (1, ))
    assert_size_stride(arg257_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg258_1, (768, 768), (768, 1))
    assert_size_stride(arg259_1, (768, ), (1, ))
    assert_size_stride(arg260_1, (768, ), (1, ))
    assert_size_stride(arg261_1, (768, ), (1, ))
    assert_size_stride(arg262_1, (768, ), (1, ))
    assert_size_stride(arg263_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg264_1, (768, ), (1, ))
    assert_size_stride(arg265_1, (768, ), (1, ))
    assert_size_stride(arg266_1, (768, ), (1, ))
    assert_size_stride(arg267_1, (768, ), (1, ))
    assert_size_stride(arg268_1, (768, ), (1, ))
    assert_size_stride(arg269_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg270_1, (768, ), (1, ))
    assert_size_stride(arg271_1, (768, ), (1, ))
    assert_size_stride(arg272_1, (768, ), (1, ))
    assert_size_stride(arg273_1, (768, ), (1, ))
    assert_size_stride(arg274_1, (3072, 768), (768, 1))
    assert_size_stride(arg275_1, (3072, ), (1, ))
    assert_size_stride(arg276_1, (768, 3072), (3072, 1))
    assert_size_stride(arg277_1, (768, ), (1, ))
    assert_size_stride(arg278_1, (768, ), (1, ))
    assert_size_stride(arg279_1, (768, ), (1, ))
    assert_size_stride(arg280_1, (768, ), (1, ))
    assert_size_stride(arg281_1, (2304, 768), (768, 1))
    assert_size_stride(arg282_1, (2304, ), (1, ))
    assert_size_stride(arg283_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg284_1, (768, 768), (768, 1))
    assert_size_stride(arg285_1, (768, ), (1, ))
    assert_size_stride(arg286_1, (768, ), (1, ))
    assert_size_stride(arg287_1, (768, ), (1, ))
    assert_size_stride(arg288_1, (768, ), (1, ))
    assert_size_stride(arg289_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg290_1, (768, ), (1, ))
    assert_size_stride(arg291_1, (768, ), (1, ))
    assert_size_stride(arg292_1, (768, ), (1, ))
    assert_size_stride(arg293_1, (768, ), (1, ))
    assert_size_stride(arg294_1, (768, ), (1, ))
    assert_size_stride(arg295_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg296_1, (768, ), (1, ))
    assert_size_stride(arg297_1, (768, ), (1, ))
    assert_size_stride(arg298_1, (768, ), (1, ))
    assert_size_stride(arg299_1, (768, ), (1, ))
    assert_size_stride(arg300_1, (3072, 768), (768, 1))
    assert_size_stride(arg301_1, (3072, ), (1, ))
    assert_size_stride(arg302_1, (768, 3072), (3072, 1))
    assert_size_stride(arg303_1, (768, ), (1, ))
    assert_size_stride(arg304_1, (768, ), (1, ))
    assert_size_stride(arg305_1, (768, ), (1, ))
    assert_size_stride(arg306_1, (768, ), (1, ))
    assert_size_stride(arg307_1, (2304, 768), (768, 1))
    assert_size_stride(arg308_1, (2304, ), (1, ))
    assert_size_stride(arg309_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg310_1, (768, 768), (768, 1))
    assert_size_stride(arg311_1, (768, ), (1, ))
    assert_size_stride(arg312_1, (768, ), (1, ))
    assert_size_stride(arg313_1, (768, ), (1, ))
    assert_size_stride(arg314_1, (768, ), (1, ))
    assert_size_stride(arg315_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg316_1, (768, ), (1, ))
    assert_size_stride(arg317_1, (768, ), (1, ))
    assert_size_stride(arg318_1, (768, ), (1, ))
    assert_size_stride(arg319_1, (768, ), (1, ))
    assert_size_stride(arg320_1, (768, ), (1, ))
    assert_size_stride(arg321_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg322_1, (768, ), (1, ))
    assert_size_stride(arg323_1, (768, ), (1, ))
    assert_size_stride(arg324_1, (768, ), (1, ))
    assert_size_stride(arg325_1, (768, ), (1, ))
    assert_size_stride(arg326_1, (3072, 768), (768, 1))
    assert_size_stride(arg327_1, (3072, ), (1, ))
    assert_size_stride(arg328_1, (768, 3072), (3072, 1))
    assert_size_stride(arg329_1, (768, ), (1, ))
    assert_size_stride(arg330_1, (768, ), (1, ))
    assert_size_stride(arg331_1, (768, ), (1, ))
    assert_size_stride(arg332_1, (768, ), (1, ))
    assert_size_stride(arg333_1, (2304, 768), (768, 1))
    assert_size_stride(arg334_1, (2304, ), (1, ))
    assert_size_stride(arg335_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg336_1, (768, 768), (768, 1))
    assert_size_stride(arg337_1, (768, ), (1, ))
    assert_size_stride(arg338_1, (768, ), (1, ))
    assert_size_stride(arg339_1, (768, ), (1, ))
    assert_size_stride(arg340_1, (768, ), (1, ))
    assert_size_stride(arg341_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg342_1, (768, ), (1, ))
    assert_size_stride(arg343_1, (768, ), (1, ))
    assert_size_stride(arg344_1, (768, ), (1, ))
    assert_size_stride(arg345_1, (768, ), (1, ))
    assert_size_stride(arg346_1, (768, ), (1, ))
    assert_size_stride(arg347_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg348_1, (768, ), (1, ))
    assert_size_stride(arg349_1, (768, ), (1, ))
    assert_size_stride(arg350_1, (768, ), (1, ))
    assert_size_stride(arg351_1, (768, ), (1, ))
    assert_size_stride(arg352_1, (3072, 768), (768, 1))
    assert_size_stride(arg353_1, (3072, ), (1, ))
    assert_size_stride(arg354_1, (768, 3072), (3072, 1))
    assert_size_stride(arg355_1, (768, ), (1, ))
    assert_size_stride(arg356_1, (768, ), (1, ))
    assert_size_stride(arg357_1, (768, ), (1, ))
    assert_size_stride(arg358_1, (768, ), (1, ))
    assert_size_stride(arg359_1, (2304, 768), (768, 1))
    assert_size_stride(arg360_1, (2304, ), (1, ))
    assert_size_stride(arg361_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg362_1, (768, 768), (768, 1))
    assert_size_stride(arg363_1, (768, ), (1, ))
    assert_size_stride(arg364_1, (768, ), (1, ))
    assert_size_stride(arg365_1, (768, ), (1, ))
    assert_size_stride(arg366_1, (768, ), (1, ))
    assert_size_stride(arg367_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg368_1, (768, ), (1, ))
    assert_size_stride(arg369_1, (768, ), (1, ))
    assert_size_stride(arg370_1, (768, ), (1, ))
    assert_size_stride(arg371_1, (768, ), (1, ))
    assert_size_stride(arg372_1, (768, ), (1, ))
    assert_size_stride(arg373_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg374_1, (768, ), (1, ))
    assert_size_stride(arg375_1, (768, ), (1, ))
    assert_size_stride(arg376_1, (768, ), (1, ))
    assert_size_stride(arg377_1, (768, ), (1, ))
    assert_size_stride(arg378_1, (3072, 768), (768, 1))
    assert_size_stride(arg379_1, (3072, ), (1, ))
    assert_size_stride(arg380_1, (768, 3072), (3072, 1))
    assert_size_stride(arg381_1, (768, ), (1, ))
    assert_size_stride(arg382_1, (768, ), (1, ))
    assert_size_stride(arg383_1, (768, ), (1, ))
    assert_size_stride(arg384_1, (768, ), (1, ))
    assert_size_stride(arg385_1, (2304, 768), (768, 1))
    assert_size_stride(arg386_1, (2304, ), (1, ))
    assert_size_stride(arg387_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg388_1, (768, 768), (768, 1))
    assert_size_stride(arg389_1, (768, ), (1, ))
    assert_size_stride(arg390_1, (768, ), (1, ))
    assert_size_stride(arg391_1, (768, ), (1, ))
    assert_size_stride(arg392_1, (768, ), (1, ))
    assert_size_stride(arg393_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg394_1, (768, ), (1, ))
    assert_size_stride(arg395_1, (768, ), (1, ))
    assert_size_stride(arg396_1, (768, ), (1, ))
    assert_size_stride(arg397_1, (768, ), (1, ))
    assert_size_stride(arg398_1, (768, ), (1, ))
    assert_size_stride(arg399_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg400_1, (768, ), (1, ))
    assert_size_stride(arg401_1, (768, ), (1, ))
    assert_size_stride(arg402_1, (768, ), (1, ))
    assert_size_stride(arg403_1, (768, ), (1, ))
    assert_size_stride(arg404_1, (3072, 768), (768, 1))
    assert_size_stride(arg405_1, (3072, ), (1, ))
    assert_size_stride(arg406_1, (768, 3072), (3072, 1))
    assert_size_stride(arg407_1, (768, ), (1, ))
    assert_size_stride(arg408_1, (768, ), (1, ))
    assert_size_stride(arg409_1, (768, ), (1, ))
    assert_size_stride(arg410_1, (768, ), (1, ))
    assert_size_stride(arg411_1, (2304, 768), (768, 1))
    assert_size_stride(arg412_1, (2304, ), (1, ))
    assert_size_stride(arg413_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg414_1, (768, 768), (768, 1))
    assert_size_stride(arg415_1, (768, ), (1, ))
    assert_size_stride(arg416_1, (768, ), (1, ))
    assert_size_stride(arg417_1, (768, ), (1, ))
    assert_size_stride(arg418_1, (768, ), (1, ))
    assert_size_stride(arg419_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg420_1, (768, ), (1, ))
    assert_size_stride(arg421_1, (768, ), (1, ))
    assert_size_stride(arg422_1, (768, ), (1, ))
    assert_size_stride(arg423_1, (768, ), (1, ))
    assert_size_stride(arg424_1, (768, ), (1, ))
    assert_size_stride(arg425_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg426_1, (768, ), (1, ))
    assert_size_stride(arg427_1, (768, ), (1, ))
    assert_size_stride(arg428_1, (768, ), (1, ))
    assert_size_stride(arg429_1, (768, ), (1, ))
    assert_size_stride(arg430_1, (3072, 768), (768, 1))
    assert_size_stride(arg431_1, (3072, ), (1, ))
    assert_size_stride(arg432_1, (768, 3072), (3072, 1))
    assert_size_stride(arg433_1, (768, ), (1, ))
    assert_size_stride(arg434_1, (768, ), (1, ))
    assert_size_stride(arg435_1, (768, ), (1, ))
    assert_size_stride(arg436_1, (768, ), (1, ))
    assert_size_stride(arg437_1, (2304, 768), (768, 1))
    assert_size_stride(arg438_1, (2304, ), (1, ))
    assert_size_stride(arg439_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg440_1, (768, 768), (768, 1))
    assert_size_stride(arg441_1, (768, ), (1, ))
    assert_size_stride(arg442_1, (768, ), (1, ))
    assert_size_stride(arg443_1, (768, ), (1, ))
    assert_size_stride(arg444_1, (768, ), (1, ))
    assert_size_stride(arg445_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg446_1, (768, ), (1, ))
    assert_size_stride(arg447_1, (768, ), (1, ))
    assert_size_stride(arg448_1, (768, ), (1, ))
    assert_size_stride(arg449_1, (768, ), (1, ))
    assert_size_stride(arg450_1, (768, ), (1, ))
    assert_size_stride(arg451_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg452_1, (768, ), (1, ))
    assert_size_stride(arg453_1, (768, ), (1, ))
    assert_size_stride(arg454_1, (768, ), (1, ))
    assert_size_stride(arg455_1, (768, ), (1, ))
    assert_size_stride(arg456_1, (3072, 768), (768, 1))
    assert_size_stride(arg457_1, (3072, ), (1, ))
    assert_size_stride(arg458_1, (768, 3072), (3072, 1))
    assert_size_stride(arg459_1, (768, ), (1, ))
    assert_size_stride(arg460_1, (768, ), (1, ))
    assert_size_stride(arg461_1, (768, ), (1, ))
    assert_size_stride(arg462_1, (768, ), (1, ))
    assert_size_stride(arg463_1, (2304, 768), (768, 1))
    assert_size_stride(arg464_1, (2304, ), (1, ))
    assert_size_stride(arg465_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg466_1, (768, 768), (768, 1))
    assert_size_stride(arg467_1, (768, ), (1, ))
    assert_size_stride(arg468_1, (768, ), (1, ))
    assert_size_stride(arg469_1, (768, ), (1, ))
    assert_size_stride(arg470_1, (768, ), (1, ))
    assert_size_stride(arg471_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg472_1, (768, ), (1, ))
    assert_size_stride(arg473_1, (768, ), (1, ))
    assert_size_stride(arg474_1, (768, ), (1, ))
    assert_size_stride(arg475_1, (768, ), (1, ))
    assert_size_stride(arg476_1, (768, ), (1, ))
    assert_size_stride(arg477_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg478_1, (768, ), (1, ))
    assert_size_stride(arg479_1, (768, ), (1, ))
    assert_size_stride(arg480_1, (768, ), (1, ))
    assert_size_stride(arg481_1, (768, ), (1, ))
    assert_size_stride(arg482_1, (3072, 768), (768, 1))
    assert_size_stride(arg483_1, (3072, ), (1, ))
    assert_size_stride(arg484_1, (768, 3072), (3072, 1))
    assert_size_stride(arg485_1, (768, ), (1, ))
    assert_size_stride(arg486_1, (768, ), (1, ))
    assert_size_stride(arg487_1, (768, ), (1, ))
    assert_size_stride(arg488_1, (768, ), (1, ))
    assert_size_stride(arg489_1, (2304, 768), (768, 1))
    assert_size_stride(arg490_1, (2304, ), (1, ))
    assert_size_stride(arg491_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg492_1, (768, 768), (768, 1))
    assert_size_stride(arg493_1, (768, ), (1, ))
    assert_size_stride(arg494_1, (768, ), (1, ))
    assert_size_stride(arg495_1, (768, ), (1, ))
    assert_size_stride(arg496_1, (768, ), (1, ))
    assert_size_stride(arg497_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg498_1, (768, ), (1, ))
    assert_size_stride(arg499_1, (768, ), (1, ))
    assert_size_stride(arg500_1, (768, ), (1, ))
    assert_size_stride(arg501_1, (768, ), (1, ))
    assert_size_stride(arg502_1, (768, ), (1, ))
    assert_size_stride(arg503_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg504_1, (768, ), (1, ))
    assert_size_stride(arg505_1, (768, ), (1, ))
    assert_size_stride(arg506_1, (768, ), (1, ))
    assert_size_stride(arg507_1, (768, ), (1, ))
    assert_size_stride(arg508_1, (3072, 768), (768, 1))
    assert_size_stride(arg509_1, (3072, ), (1, ))
    assert_size_stride(arg510_1, (768, 3072), (3072, 1))
    assert_size_stride(arg511_1, (768, ), (1, ))
    assert_size_stride(arg512_1, (768, ), (1, ))
    assert_size_stride(arg513_1, (768, ), (1, ))
    assert_size_stride(arg514_1, (768, ), (1, ))
    assert_size_stride(arg515_1, (2304, 768), (768, 1))
    assert_size_stride(arg516_1, (2304, ), (1, ))
    assert_size_stride(arg517_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg518_1, (768, 768), (768, 1))
    assert_size_stride(arg519_1, (768, ), (1, ))
    assert_size_stride(arg520_1, (768, ), (1, ))
    assert_size_stride(arg521_1, (768, ), (1, ))
    assert_size_stride(arg522_1, (768, ), (1, ))
    assert_size_stride(arg523_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg524_1, (768, ), (1, ))
    assert_size_stride(arg525_1, (768, ), (1, ))
    assert_size_stride(arg526_1, (768, ), (1, ))
    assert_size_stride(arg527_1, (768, ), (1, ))
    assert_size_stride(arg528_1, (768, ), (1, ))
    assert_size_stride(arg529_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg530_1, (768, ), (1, ))
    assert_size_stride(arg531_1, (768, ), (1, ))
    assert_size_stride(arg532_1, (768, ), (1, ))
    assert_size_stride(arg533_1, (768, ), (1, ))
    assert_size_stride(arg534_1, (3072, 768), (768, 1))
    assert_size_stride(arg535_1, (3072, ), (1, ))
    assert_size_stride(arg536_1, (768, 3072), (3072, 1))
    assert_size_stride(arg537_1, (768, ), (1, ))
    assert_size_stride(arg538_1, (768, ), (1, ))
    assert_size_stride(arg539_1, (768, ), (1, ))
    assert_size_stride(arg540_1, (768, ), (1, ))
    assert_size_stride(arg541_1, (2304, 768), (768, 1))
    assert_size_stride(arg542_1, (2304, ), (1, ))
    assert_size_stride(arg543_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg544_1, (768, 768), (768, 1))
    assert_size_stride(arg545_1, (768, ), (1, ))
    assert_size_stride(arg546_1, (768, ), (1, ))
    assert_size_stride(arg547_1, (768, ), (1, ))
    assert_size_stride(arg548_1, (768, ), (1, ))
    assert_size_stride(arg549_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg550_1, (768, ), (1, ))
    assert_size_stride(arg551_1, (768, ), (1, ))
    assert_size_stride(arg552_1, (768, ), (1, ))
    assert_size_stride(arg553_1, (768, ), (1, ))
    assert_size_stride(arg554_1, (768, ), (1, ))
    assert_size_stride(arg555_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg556_1, (768, ), (1, ))
    assert_size_stride(arg557_1, (768, ), (1, ))
    assert_size_stride(arg558_1, (768, ), (1, ))
    assert_size_stride(arg559_1, (768, ), (1, ))
    assert_size_stride(arg560_1, (3072, 768), (768, 1))
    assert_size_stride(arg561_1, (3072, ), (1, ))
    assert_size_stride(arg562_1, (768, 3072), (3072, 1))
    assert_size_stride(arg563_1, (768, ), (1, ))
    assert_size_stride(arg564_1, (768, ), (1, ))
    assert_size_stride(arg565_1, (768, ), (1, ))
    assert_size_stride(arg566_1, (768, ), (1, ))
    assert_size_stride(arg567_1, (2304, 768), (768, 1))
    assert_size_stride(arg568_1, (2304, ), (1, ))
    assert_size_stride(arg569_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg570_1, (768, 768), (768, 1))
    assert_size_stride(arg571_1, (768, ), (1, ))
    assert_size_stride(arg572_1, (768, ), (1, ))
    assert_size_stride(arg573_1, (768, ), (1, ))
    assert_size_stride(arg574_1, (768, ), (1, ))
    assert_size_stride(arg575_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg576_1, (768, ), (1, ))
    assert_size_stride(arg577_1, (768, ), (1, ))
    assert_size_stride(arg578_1, (768, ), (1, ))
    assert_size_stride(arg579_1, (768, ), (1, ))
    assert_size_stride(arg580_1, (768, ), (1, ))
    assert_size_stride(arg581_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg582_1, (768, ), (1, ))
    assert_size_stride(arg583_1, (768, ), (1, ))
    assert_size_stride(arg584_1, (768, ), (1, ))
    assert_size_stride(arg585_1, (768, ), (1, ))
    assert_size_stride(arg586_1, (3072, 768), (768, 1))
    assert_size_stride(arg587_1, (3072, ), (1, ))
    assert_size_stride(arg588_1, (768, 3072), (3072, 1))
    assert_size_stride(arg589_1, (768, ), (1, ))
    assert_size_stride(arg590_1, (768, ), (1, ))
    assert_size_stride(arg591_1, (768, ), (1, ))
    assert_size_stride(arg592_1, (768, ), (1, ))
    assert_size_stride(arg593_1, (2304, 768), (768, 1))
    assert_size_stride(arg594_1, (2304, ), (1, ))
    assert_size_stride(arg595_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg596_1, (768, 768), (768, 1))
    assert_size_stride(arg597_1, (768, ), (1, ))
    assert_size_stride(arg598_1, (768, ), (1, ))
    assert_size_stride(arg599_1, (768, ), (1, ))
    assert_size_stride(arg600_1, (768, ), (1, ))
    assert_size_stride(arg601_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg602_1, (768, ), (1, ))
    assert_size_stride(arg603_1, (768, ), (1, ))
    assert_size_stride(arg604_1, (768, ), (1, ))
    assert_size_stride(arg605_1, (768, ), (1, ))
    assert_size_stride(arg606_1, (768, ), (1, ))
    assert_size_stride(arg607_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg608_1, (768, ), (1, ))
    assert_size_stride(arg609_1, (768, ), (1, ))
    assert_size_stride(arg610_1, (768, ), (1, ))
    assert_size_stride(arg611_1, (768, ), (1, ))
    assert_size_stride(arg612_1, (3072, 768), (768, 1))
    assert_size_stride(arg613_1, (3072, ), (1, ))
    assert_size_stride(arg614_1, (768, 3072), (3072, 1))
    assert_size_stride(arg615_1, (768, ), (1, ))
    assert_size_stride(arg616_1, (768, ), (1, ))
    assert_size_stride(arg617_1, (768, ), (1, ))
    assert_size_stride(arg618_1, (768, ), (1, ))
    assert_size_stride(arg619_1, (2304, 768), (768, 1))
    assert_size_stride(arg620_1, (2304, ), (1, ))
    assert_size_stride(arg621_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(arg622_1, (768, 768), (768, 1))
    assert_size_stride(arg623_1, (768, ), (1, ))
    assert_size_stride(arg624_1, (768, ), (1, ))
    assert_size_stride(arg625_1, (768, ), (1, ))
    assert_size_stride(arg626_1, (768, ), (1, ))
    assert_size_stride(arg627_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg628_1, (768, ), (1, ))
    assert_size_stride(arg629_1, (768, ), (1, ))
    assert_size_stride(arg630_1, (768, ), (1, ))
    assert_size_stride(arg631_1, (768, ), (1, ))
    assert_size_stride(arg632_1, (768, ), (1, ))
    assert_size_stride(arg633_1, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg634_1, (768, ), (1, ))
    assert_size_stride(arg635_1, (768, ), (1, ))
    assert_size_stride(arg636_1, (768, ), (1, ))
    assert_size_stride(arg637_1, (768, ), (1, ))
    assert_size_stride(arg638_1, (3072, 768), (768, 1))
    assert_size_stride(arg639_1, (3072, ), (1, ))
    assert_size_stride(arg640_1, (768, 3072), (3072, 1))
    assert_size_stride(arg641_1, (768, ), (1, ))
    assert_size_stride(arg642_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg643_1, (768, ), (1, ))
    assert_size_stride(arg644_1, (768, ), (1, ))
    assert_size_stride(arg645_1, (768, 768), (768, 1))
    assert_size_stride(arg646_1, (768, ), (1, ))
    assert_size_stride(arg647_1, (768, 768), (768, 1))
    assert_size_stride(arg648_1, (768, ), (1, ))
    assert_size_stride(arg649_1, (768, 768), (768, 1))
    assert_size_stride(arg650_1, (768, ), (1, ))
    assert_size_stride(arg651_1, (768, 768), (768, 1))
    assert_size_stride(arg652_1, (768, ), (1, ))
    assert_size_stride(arg653_1, (768, ), (1, ))
    assert_size_stride(arg654_1, (768, ), (1, ))
    assert_size_stride(arg655_1, (768, ), (1, ))
    assert_size_stride(arg656_1, (768, ), (1, ))
    assert_size_stride(arg657_1, (3072, 768), (768, 1))
    assert_size_stride(arg658_1, (3072, ), (1, ))
    assert_size_stride(arg659_1, (768, 3072), (3072, 1))
    assert_size_stride(arg660_1, (768, ), (1, ))
    assert_size_stride(arg661_1, (768, ), (1, ))
    assert_size_stride(arg662_1, (768, ), (1, ))
    assert_size_stride(arg663_1, (768, 768), (768, 1))
    assert_size_stride(arg664_1, (768, ), (1, ))
    assert_size_stride(arg665_1, (768, 768), (768, 1))
    assert_size_stride(arg666_1, (768, ), (1, ))
    assert_size_stride(arg667_1, (768, 768), (768, 1))
    assert_size_stride(arg668_1, (768, ), (1, ))
    assert_size_stride(arg669_1, (768, 768), (768, 1))
    assert_size_stride(arg670_1, (768, ), (1, ))
    assert_size_stride(arg671_1, (768, ), (1, ))
    assert_size_stride(arg672_1, (768, ), (1, ))
    assert_size_stride(arg673_1, (768, ), (1, ))
    assert_size_stride(arg674_1, (768, ), (1, ))
    assert_size_stride(arg675_1, (3072, 768), (768, 1))
    assert_size_stride(arg676_1, (3072, ), (1, ))
    assert_size_stride(arg677_1, (768, 3072), (3072, 1))
    assert_size_stride(arg678_1, (768, ), (1, ))
    assert_size_stride(arg679_1, (768, ), (1, ))
    assert_size_stride(arg680_1, (768, ), (1, ))
    assert_size_stride(arg681_1, (1000, 768), (768, 1))
    assert_size_stride(arg682_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg0_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((192, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg1_1, buf1, 576, 9, grid=grid(576, 9), stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 192, 112, 112), (2408448, 1, 21504, 192))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((8, 192, 112, 112), (2408448, 1, 21504, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_gelu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, buf4, 19267584, grid=grid(19267584), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf3
        buf5 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.gelu, aten.convolution]
        triton_poi_fused_convolution_gelu_3.run(arg6_1, buf5, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.gelu, aten.convolution]
        buf6 = extern_kernels.convolution(buf4, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 384, 56, 56), (1204224, 1, 21504, 384))
        del buf5
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided_cuda((8, 384, 56, 56), (1204224, 1, 21504, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.gelu]
        triton_poi_fused__native_batch_norm_legit_no_training_gelu_4.run(buf7, arg7_1, arg8_1, arg9_1, arg10_1, buf8, 9633792, grid=grid(9633792), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        del buf7
        buf9 = empty_strided_cuda((768, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten.gelu, aten.convolution]
        triton_poi_fused_convolution_gelu_5.run(arg11_1, buf9, 294912, 9, grid=grid(294912, 9), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten.gelu, aten.convolution]
        buf10 = extern_kernels.convolution(buf8, buf9, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del buf8
        del buf9
        buf11 = empty_strided_cuda((1, 28, 28, 16, 2), (25088, 896, 32, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stack_3], Original ATen: [aten.stack]
        triton_poi_fused_stack_6.run(buf11, 25088, grid=grid(25088), stream=stream0)
        buf12 = empty_strided_cuda((1, 28, 28, 16, 2), (25088, 896, 32, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stack_2], Original ATen: [aten.stack]
        triton_poi_fused_stack_7.run(buf12, 25088, grid=grid(25088), stream=stream0)
        buf13 = empty_strided_cuda((1, 28, 28, 64), (50176, 1792, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
        triton_poi_fused_cat_8.run(buf11, buf12, buf13, 50176, grid=grid(50176), stream=stream0)
        del buf11
        del buf12
        # Topologically Sorted Source Nodes: [pos_3], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(reinterpret_tensor(buf13, (1, 64, 28, 28), (0, 1, 1792, 64), 0), arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (1, 768, 28, 28), (602112, 784, 28, 1))
        del arg16_1
        del buf13
        buf15 = reinterpret_tensor(buf10, (8, 784, 768), (602112, 768, 1), 0); del buf10  # reuse
        buf19 = empty_strided_cuda((8, 784, 768), (602112, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_459, layer_norm_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_9.run(buf15, arg12_1, arg13_1, arg14_1, arg15_1, buf14, arg17_1, arg19_1, arg20_1, buf19, 6272, 768, grid=grid(6272), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del arg17_1
        del arg19_1
        del arg20_1
        del buf14
        buf20 = empty_strided_cuda((6272, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (6272, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 2304), (1, 768), 0), out=buf20)
        del arg21_1
        buf21 = empty_strided_cuda((8, 16, 48, 1, 7), (5376, 48, 1, 43008, 768), torch.float32)
        # Topologically Sorted Source Nodes: [q_51], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf20, arg22_1, buf21, 43008, 112, grid=grid(43008), stream=stream0)
        buf22 = empty_strided_cuda((8, 16, 48, 1), (768, 48, 1, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [q_51], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf21, buf22, 6144, 7, grid=grid(6144), stream=stream0)
        buf23 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [k_51], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf20, arg22_1, buf23, 43008, 112, grid=grid(43008), stream=stream0)
        buf24 = empty_strided_cuda((8, 16, 48, 1), (768, 48, 1, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [k_51], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf23, buf24, 6144, 7, grid=grid(6144), stream=stream0)
        buf25 = reinterpret_tensor(buf19, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [q_51, matmul_48], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf20, arg22_1, buf22, buf25, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf26 = empty_strided_cuda((8, 16, 784, 48), (602112, 37632, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf20, arg22_1, buf24, buf26, 4816896, grid=grid(4816896), stream=stream0)
        buf27 = empty_strided_cuda((128, 48, 48), (2304, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf25, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf26, (128, 784, 48), (37632, 48, 1), 0), out=buf27)
        buf30 = empty_strided_cuda((8, 16, 48, 48), (36864, 2304, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_73], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf27, arg23_1, buf30, 6144, 48, grid=grid(6144), stream=stream0)
        del arg23_1
        buf31 = reinterpret_tensor(buf26, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_461], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf20, arg22_1, buf31, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg22_1
        buf32 = reinterpret_tensor(buf25, (128, 48, 784), (37632, 784, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_461], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf30, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf31, (128, 48, 784), (37632, 784, 1), 0), out=buf32)
        buf33 = reinterpret_tensor(buf31, (8, 784, 768), (602112, 768, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_463], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf32, buf33, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf34 = reinterpret_tensor(buf32, (6272, 768), (768, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_463], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (6272, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 768), (1, 768), 0), out=buf34)
        del arg24_1
        buf38 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_463, mul_107, x_465, layer_norm_78], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf15, arg18_1, buf34, arg25_1, arg27_1, arg28_1, buf38, 6272, 768, grid=grid(6272), stream=stream0)
        del arg27_1
        del arg28_1
        # Topologically Sorted Source Nodes: [x_467], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(reinterpret_tensor(buf38, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg29_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf39, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg29_1
        del buf38
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_467, x_468, x_469], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf40, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg30_1
        del arg31_1
        del arg32_1
        del arg33_1
        del arg34_1
        # Topologically Sorted Source Nodes: [x_467, x_468, x_469, x_470], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf41 = extern_kernels.convolution(buf40, arg35_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf41, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg35_1
        buf42 = reinterpret_tensor(buf41, (8, 784, 768), (602112, 768, 1), 0); del buf41  # reuse
        buf46 = reinterpret_tensor(buf40, (8, 784, 768), (602112, 768, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_463, mul_107, x_465, mul_108, x_472, layer_norm_79], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf42, buf15, arg18_1, buf34, arg25_1, arg26_1, arg36_1, arg38_1, arg39_1, buf46, 6272, 768, grid=grid(6272), stream=stream0)
        del arg18_1
        del arg25_1
        del arg26_1
        del arg36_1
        del arg38_1
        del arg39_1
        buf47 = reinterpret_tensor(buf4, (6272, 3072), (3072, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (6272, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 3072), (1, 768), 0), out=buf47)
        del arg40_1
        buf48 = reinterpret_tensor(buf47, (8, 784, 3072), (2408448, 3072, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_474], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf48, arg41_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg41_1
        buf49 = reinterpret_tensor(buf46, (6272, 768), (768, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf48, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg42_1, (3072, 768), (1, 3072), 0), out=buf49)
        del arg42_1
        buf53 = reinterpret_tensor(buf34, (8, 784, 768), (602112, 768, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [mul_109, x_478, layer_norm_80], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf42, arg37_1, buf49, arg43_1, arg45_1, arg46_1, buf53, 6272, 768, grid=grid(6272), stream=stream0)
        del arg45_1
        del arg46_1
        buf54 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (6272, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 2304), (1, 768), 0), out=buf54)
        del arg47_1
        buf55 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [q_53], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf54, arg48_1, buf55, 43008, 112, grid=grid(43008), stream=stream0)
        buf56 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [q_53], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf55, buf56, 6144, 7, grid=grid(6144), stream=stream0)
        buf57 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [k_53], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf54, arg48_1, buf57, 43008, 112, grid=grid(43008), stream=stream0)
        buf58 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [k_53], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf57, buf58, 6144, 7, grid=grid(6144), stream=stream0)
        buf59 = reinterpret_tensor(buf53, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [q_53, matmul_50], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf54, arg48_1, buf56, buf59, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf60 = reinterpret_tensor(buf15, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [matmul_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf54, arg48_1, buf58, buf60, 4816896, grid=grid(4816896), stream=stream0)
        buf61 = reinterpret_tensor(buf30, (128, 48, 48), (2304, 48, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [matmul_50], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf60, (128, 784, 48), (37632, 48, 1), 0), out=buf61)
        buf64 = reinterpret_tensor(buf27, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [attn_76], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf61, arg49_1, buf64, 6144, 48, grid=grid(6144), stream=stream0)
        del arg49_1
        buf65 = reinterpret_tensor(buf60, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_479], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf54, arg48_1, buf65, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg48_1
        buf66 = reinterpret_tensor(buf59, (128, 48, 784), (37632, 784, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_479], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf64, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf65, (128, 48, 784), (37632, 784, 1), 0), out=buf66)
        buf67 = reinterpret_tensor(buf65, (8, 784, 768), (602112, 768, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_481], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf66, buf67, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf68 = reinterpret_tensor(buf66, (6272, 768), (768, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_481], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (6272, 768), (768, 1), 0), reinterpret_tensor(arg50_1, (768, 768), (1, 768), 0), out=buf68)
        del arg50_1
        buf69 = reinterpret_tensor(buf68, (8, 784, 768), (602112, 768, 1), 0); del buf68  # reuse
        buf73 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [mul_109, x_478, x_481, mul_111, x_483, layer_norm_81], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf69, buf42, arg37_1, buf49, arg43_1, arg44_1, arg51_1, arg53_1, arg54_1, buf73, 6272, 768, grid=grid(6272), stream=stream0)
        del arg37_1
        del arg43_1
        del arg44_1
        del arg51_1
        del arg53_1
        del arg54_1
        del buf42
        del buf49
        # Topologically Sorted Source Nodes: [x_485], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(reinterpret_tensor(buf73, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg55_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf74, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg55_1
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_485, x_486, x_487], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf75, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg56_1
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        # Topologically Sorted Source Nodes: [x_485, x_486, x_487, x_488], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf76 = extern_kernels.convolution(buf75, arg61_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf76, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg61_1
        buf80 = reinterpret_tensor(buf75, (8, 784, 768), (602112, 768, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [mul_112, x_490, layer_norm_82], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf69, arg52_1, buf76, arg62_1, arg64_1, arg65_1, buf80, 6272, 768, grid=grid(6272), stream=stream0)
        del arg64_1
        del arg65_1
        buf81 = reinterpret_tensor(buf48, (6272, 3072), (3072, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (6272, 768), (768, 1), 0), reinterpret_tensor(arg66_1, (768, 3072), (1, 768), 0), out=buf81)
        del arg66_1
        buf82 = reinterpret_tensor(buf81, (8, 784, 3072), (2408448, 3072, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_492], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf82, arg67_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg67_1
        buf83 = reinterpret_tensor(buf80, (6272, 768), (768, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg68_1, (3072, 768), (1, 3072), 0), out=buf83)
        del arg68_1
        buf84 = reinterpret_tensor(buf83, (8, 784, 768), (602112, 768, 1), 0); del buf83  # reuse
        buf88 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [mul_112, x_490, mul_113, x_496, layer_norm_83], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf84, buf69, arg52_1, buf76, arg62_1, arg63_1, arg69_1, arg71_1, arg72_1, buf88, 6272, 768, grid=grid(6272), stream=stream0)
        del arg52_1
        del arg62_1
        del arg63_1
        del arg69_1
        del arg71_1
        del arg72_1
        del buf69
        buf89 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf88, (6272, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 2304), (1, 768), 0), out=buf89)
        del arg73_1
        buf90 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [q_55], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf89, arg74_1, buf90, 43008, 112, grid=grid(43008), stream=stream0)
        buf91 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [q_55], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf90, buf91, 6144, 7, grid=grid(6144), stream=stream0)
        buf92 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [k_55], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf89, arg74_1, buf92, 43008, 112, grid=grid(43008), stream=stream0)
        buf93 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [k_55], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf92, buf93, 6144, 7, grid=grid(6144), stream=stream0)
        buf94 = reinterpret_tensor(buf88, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [q_55, matmul_52], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf89, arg74_1, buf91, buf94, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf95 = reinterpret_tensor(buf76, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [matmul_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf89, arg74_1, buf93, buf95, 4816896, grid=grid(4816896), stream=stream0)
        buf96 = reinterpret_tensor(buf64, (128, 48, 48), (2304, 48, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [matmul_52], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf95, (128, 784, 48), (37632, 48, 1), 0), out=buf96)
        buf99 = reinterpret_tensor(buf61, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [attn_79], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf96, arg75_1, buf99, 6144, 48, grid=grid(6144), stream=stream0)
        del arg75_1
        buf100 = reinterpret_tensor(buf95, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [x_497], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf89, arg74_1, buf100, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg74_1
        buf101 = reinterpret_tensor(buf94, (128, 48, 784), (37632, 784, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_497], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf99, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf100, (128, 48, 784), (37632, 784, 1), 0), out=buf101)
        buf102 = reinterpret_tensor(buf100, (8, 784, 768), (602112, 768, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_499], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf101, buf102, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf103 = reinterpret_tensor(buf101, (6272, 768), (768, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_499], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (6272, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 768), (1, 768), 0), out=buf103)
        del arg76_1
        buf107 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_499, mul_115, x_501, layer_norm_84], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf84, arg70_1, buf103, arg77_1, arg79_1, arg80_1, buf107, 6272, 768, grid=grid(6272), stream=stream0)
        del arg79_1
        del arg80_1
        # Topologically Sorted Source Nodes: [x_503], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(reinterpret_tensor(buf107, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg81_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf108, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg81_1
        del buf107
        buf109 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_503, x_504, x_505], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf109, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        del arg86_1
        # Topologically Sorted Source Nodes: [x_503, x_504, x_505, x_506], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf110 = extern_kernels.convolution(buf109, arg87_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf110, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg87_1
        buf111 = reinterpret_tensor(buf110, (8, 784, 768), (602112, 768, 1), 0); del buf110  # reuse
        buf115 = reinterpret_tensor(buf109, (8, 784, 768), (602112, 768, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_499, mul_115, x_501, mul_116, x_508, layer_norm_85], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf111, buf84, arg70_1, buf103, arg77_1, arg78_1, arg88_1, arg90_1, arg91_1, buf115, 6272, 768, grid=grid(6272), stream=stream0)
        del arg70_1
        del arg77_1
        del arg78_1
        del arg88_1
        del arg90_1
        del arg91_1
        buf116 = reinterpret_tensor(buf82, (6272, 3072), (3072, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf115, (6272, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 3072), (1, 768), 0), out=buf116)
        del arg92_1
        buf117 = reinterpret_tensor(buf116, (8, 784, 3072), (2408448, 3072, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_510], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf117, arg93_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg93_1
        buf118 = reinterpret_tensor(buf115, (6272, 768), (768, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf117, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg94_1, (3072, 768), (1, 3072), 0), out=buf118)
        del arg94_1
        buf122 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [mul_117, x_514, layer_norm_86], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf111, arg89_1, buf118, arg95_1, arg97_1, arg98_1, buf122, 6272, 768, grid=grid(6272), stream=stream0)
        del arg97_1
        del arg98_1
        buf123 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (6272, 768), (768, 1), 0), reinterpret_tensor(arg99_1, (768, 2304), (1, 768), 0), out=buf123)
        del arg99_1
        buf124 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [q_57], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf123, arg100_1, buf124, 43008, 112, grid=grid(43008), stream=stream0)
        buf125 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [q_57], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf124, buf125, 6144, 7, grid=grid(6144), stream=stream0)
        buf126 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [k_57], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf123, arg100_1, buf126, 43008, 112, grid=grid(43008), stream=stream0)
        buf127 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [k_57], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf126, buf127, 6144, 7, grid=grid(6144), stream=stream0)
        buf128 = reinterpret_tensor(buf122, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [q_57, matmul_54], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf123, arg100_1, buf125, buf128, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf129 = reinterpret_tensor(buf103, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [matmul_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf123, arg100_1, buf127, buf129, 4816896, grid=grid(4816896), stream=stream0)
        buf130 = reinterpret_tensor(buf99, (128, 48, 48), (2304, 48, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [matmul_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf128, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf129, (128, 784, 48), (37632, 48, 1), 0), out=buf130)
        buf133 = reinterpret_tensor(buf96, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [attn_82], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf130, arg101_1, buf133, 6144, 48, grid=grid(6144), stream=stream0)
        del arg101_1
        buf134 = reinterpret_tensor(buf129, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [x_515], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf123, arg100_1, buf134, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg100_1
        buf135 = reinterpret_tensor(buf128, (128, 48, 784), (37632, 784, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_515], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf133, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf134, (128, 48, 784), (37632, 784, 1), 0), out=buf135)
        buf136 = reinterpret_tensor(buf134, (8, 784, 768), (602112, 768, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [x_517], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf135, buf136, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf137 = reinterpret_tensor(buf135, (6272, 768), (768, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [x_517], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (6272, 768), (768, 1), 0), reinterpret_tensor(arg102_1, (768, 768), (1, 768), 0), out=buf137)
        del arg102_1
        buf138 = reinterpret_tensor(buf137, (8, 784, 768), (602112, 768, 1), 0); del buf137  # reuse
        buf142 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [mul_117, x_514, x_517, mul_119, x_519, layer_norm_87], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf138, buf111, arg89_1, buf118, arg95_1, arg96_1, arg103_1, arg105_1, arg106_1, buf142, 6272, 768, grid=grid(6272), stream=stream0)
        del arg103_1
        del arg105_1
        del arg106_1
        del arg89_1
        del arg95_1
        del arg96_1
        del buf111
        del buf118
        # Topologically Sorted Source Nodes: [x_521], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(reinterpret_tensor(buf142, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg107_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf143, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg107_1
        buf144 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_521, x_522, x_523], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf144, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg108_1
        del arg109_1
        del arg110_1
        del arg111_1
        del arg112_1
        # Topologically Sorted Source Nodes: [x_521, x_522, x_523, x_524], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf145 = extern_kernels.convolution(buf144, arg113_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf145, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg113_1
        buf149 = reinterpret_tensor(buf144, (8, 784, 768), (602112, 768, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [mul_120, x_526, layer_norm_88], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf138, arg104_1, buf145, arg114_1, arg116_1, arg117_1, buf149, 6272, 768, grid=grid(6272), stream=stream0)
        del arg116_1
        del arg117_1
        buf150 = reinterpret_tensor(buf117, (6272, 3072), (3072, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (6272, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 3072), (1, 768), 0), out=buf150)
        del arg118_1
        buf151 = reinterpret_tensor(buf150, (8, 784, 3072), (2408448, 3072, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [x_528], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf151, arg119_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg119_1
        buf152 = reinterpret_tensor(buf149, (6272, 768), (768, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg120_1, (3072, 768), (1, 3072), 0), out=buf152)
        del arg120_1
        buf153 = reinterpret_tensor(buf152, (8, 784, 768), (602112, 768, 1), 0); del buf152  # reuse
        buf157 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [mul_120, x_526, mul_121, x_532, layer_norm_89], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf153, buf138, arg104_1, buf145, arg114_1, arg115_1, arg121_1, arg123_1, arg124_1, buf157, 6272, 768, grid=grid(6272), stream=stream0)
        del arg104_1
        del arg114_1
        del arg115_1
        del arg121_1
        del arg123_1
        del arg124_1
        del buf138
        buf158 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (6272, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 2304), (1, 768), 0), out=buf158)
        del arg125_1
        buf159 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [q_59], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf158, arg126_1, buf159, 43008, 112, grid=grid(43008), stream=stream0)
        buf160 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [q_59], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf159, buf160, 6144, 7, grid=grid(6144), stream=stream0)
        buf161 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [k_59], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf158, arg126_1, buf161, 43008, 112, grid=grid(43008), stream=stream0)
        buf162 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [k_59], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf161, buf162, 6144, 7, grid=grid(6144), stream=stream0)
        buf163 = reinterpret_tensor(buf157, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [q_59, matmul_56], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf158, arg126_1, buf160, buf163, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf164 = reinterpret_tensor(buf145, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [matmul_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf158, arg126_1, buf162, buf164, 4816896, grid=grid(4816896), stream=stream0)
        buf165 = reinterpret_tensor(buf133, (128, 48, 48), (2304, 48, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [matmul_56], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf163, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf164, (128, 784, 48), (37632, 48, 1), 0), out=buf165)
        buf168 = reinterpret_tensor(buf130, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [attn_85], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf165, arg127_1, buf168, 6144, 48, grid=grid(6144), stream=stream0)
        del arg127_1
        buf169 = reinterpret_tensor(buf164, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [x_533], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf158, arg126_1, buf169, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg126_1
        buf170 = reinterpret_tensor(buf163, (128, 48, 784), (37632, 784, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_533], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf168, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf169, (128, 48, 784), (37632, 784, 1), 0), out=buf170)
        buf171 = reinterpret_tensor(buf169, (8, 784, 768), (602112, 768, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_535], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf170, buf171, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf172 = reinterpret_tensor(buf170, (6272, 768), (768, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [x_535], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (6272, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 768), (1, 768), 0), out=buf172)
        del arg128_1
        buf176 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_535, mul_123, x_537, layer_norm_90], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf153, arg122_1, buf172, arg129_1, arg131_1, arg132_1, buf176, 6272, 768, grid=grid(6272), stream=stream0)
        del arg131_1
        del arg132_1
        # Topologically Sorted Source Nodes: [x_539], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(reinterpret_tensor(buf176, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg133_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf177, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg133_1
        del buf176
        buf178 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_539, x_540, x_541], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf178, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg134_1
        del arg135_1
        del arg136_1
        del arg137_1
        del arg138_1
        # Topologically Sorted Source Nodes: [x_539, x_540, x_541, x_542], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf179 = extern_kernels.convolution(buf178, arg139_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf179, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg139_1
        buf180 = reinterpret_tensor(buf179, (8, 784, 768), (602112, 768, 1), 0); del buf179  # reuse
        buf184 = reinterpret_tensor(buf178, (8, 784, 768), (602112, 768, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_535, mul_123, x_537, mul_124, x_544, layer_norm_91], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf180, buf153, arg122_1, buf172, arg129_1, arg130_1, arg140_1, arg142_1, arg143_1, buf184, 6272, 768, grid=grid(6272), stream=stream0)
        del arg122_1
        del arg129_1
        del arg130_1
        del arg140_1
        del arg142_1
        del arg143_1
        buf185 = reinterpret_tensor(buf151, (6272, 3072), (3072, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf184, (6272, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 3072), (1, 768), 0), out=buf185)
        del arg144_1
        buf186 = reinterpret_tensor(buf185, (8, 784, 3072), (2408448, 3072, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [x_546], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf186, arg145_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg145_1
        buf187 = reinterpret_tensor(buf184, (6272, 768), (768, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf186, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg146_1, (3072, 768), (1, 3072), 0), out=buf187)
        del arg146_1
        buf191 = reinterpret_tensor(buf172, (8, 784, 768), (602112, 768, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [mul_125, x_550, layer_norm_92], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf180, arg141_1, buf187, arg147_1, arg149_1, arg150_1, buf191, 6272, 768, grid=grid(6272), stream=stream0)
        del arg149_1
        del arg150_1
        buf192 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (6272, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 2304), (1, 768), 0), out=buf192)
        del arg151_1
        buf193 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [q_61], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf192, arg152_1, buf193, 43008, 112, grid=grid(43008), stream=stream0)
        buf194 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [q_61], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf193, buf194, 6144, 7, grid=grid(6144), stream=stream0)
        buf195 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [k_61], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf192, arg152_1, buf195, 43008, 112, grid=grid(43008), stream=stream0)
        buf196 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [k_61], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf195, buf196, 6144, 7, grid=grid(6144), stream=stream0)
        buf197 = reinterpret_tensor(buf191, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [q_61, matmul_58], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf192, arg152_1, buf194, buf197, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf198 = reinterpret_tensor(buf153, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [matmul_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf192, arg152_1, buf196, buf198, 4816896, grid=grid(4816896), stream=stream0)
        buf199 = reinterpret_tensor(buf168, (128, 48, 48), (2304, 48, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [matmul_58], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf197, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf198, (128, 784, 48), (37632, 48, 1), 0), out=buf199)
        buf202 = reinterpret_tensor(buf165, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [attn_88], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf199, arg153_1, buf202, 6144, 48, grid=grid(6144), stream=stream0)
        del arg153_1
        buf203 = reinterpret_tensor(buf198, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [x_551], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf192, arg152_1, buf203, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg152_1
        buf204 = reinterpret_tensor(buf197, (128, 48, 784), (37632, 784, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [x_551], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf202, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf203, (128, 48, 784), (37632, 784, 1), 0), out=buf204)
        buf205 = reinterpret_tensor(buf203, (8, 784, 768), (602112, 768, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [x_553], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf204, buf205, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf206 = reinterpret_tensor(buf204, (6272, 768), (768, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_553], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (6272, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 768), (1, 768), 0), out=buf206)
        del arg154_1
        buf207 = reinterpret_tensor(buf206, (8, 784, 768), (602112, 768, 1), 0); del buf206  # reuse
        buf211 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [mul_125, x_550, x_553, mul_127, x_555, layer_norm_93], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf207, buf180, arg141_1, buf187, arg147_1, arg148_1, arg155_1, arg157_1, arg158_1, buf211, 6272, 768, grid=grid(6272), stream=stream0)
        del arg141_1
        del arg147_1
        del arg148_1
        del arg155_1
        del arg157_1
        del arg158_1
        del buf180
        del buf187
        # Topologically Sorted Source Nodes: [x_557], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(reinterpret_tensor(buf211, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg159_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf212, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg159_1
        buf213 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [x_557, x_558, x_559], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf213, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg160_1
        del arg161_1
        del arg162_1
        del arg163_1
        del arg164_1
        # Topologically Sorted Source Nodes: [x_557, x_558, x_559, x_560], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf214 = extern_kernels.convolution(buf213, arg165_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf214, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg165_1
        buf218 = reinterpret_tensor(buf213, (8, 784, 768), (602112, 768, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [mul_128, x_562, layer_norm_94], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf207, arg156_1, buf214, arg166_1, arg168_1, arg169_1, buf218, 6272, 768, grid=grid(6272), stream=stream0)
        del arg168_1
        del arg169_1
        buf219 = reinterpret_tensor(buf186, (6272, 3072), (3072, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (6272, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 3072), (1, 768), 0), out=buf219)
        del arg170_1
        buf220 = reinterpret_tensor(buf219, (8, 784, 3072), (2408448, 3072, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_564], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf220, arg171_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg171_1
        buf221 = reinterpret_tensor(buf218, (6272, 768), (768, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg172_1, (3072, 768), (1, 3072), 0), out=buf221)
        del arg172_1
        buf222 = reinterpret_tensor(buf221, (8, 784, 768), (602112, 768, 1), 0); del buf221  # reuse
        buf226 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [mul_128, x_562, mul_129, x_568, layer_norm_95], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf222, buf207, arg156_1, buf214, arg166_1, arg167_1, arg173_1, arg175_1, arg176_1, buf226, 6272, 768, grid=grid(6272), stream=stream0)
        del arg156_1
        del arg166_1
        del arg167_1
        del arg173_1
        del arg175_1
        del arg176_1
        del buf207
        buf227 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (6272, 768), (768, 1), 0), reinterpret_tensor(arg177_1, (768, 2304), (1, 768), 0), out=buf227)
        del arg177_1
        buf228 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [q_63], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf227, arg178_1, buf228, 43008, 112, grid=grid(43008), stream=stream0)
        buf229 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [q_63], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf228, buf229, 6144, 7, grid=grid(6144), stream=stream0)
        buf230 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [k_63], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf227, arg178_1, buf230, 43008, 112, grid=grid(43008), stream=stream0)
        buf231 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [k_63], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf230, buf231, 6144, 7, grid=grid(6144), stream=stream0)
        buf232 = reinterpret_tensor(buf226, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [q_63, matmul_60], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf227, arg178_1, buf229, buf232, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf233 = reinterpret_tensor(buf214, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [matmul_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf227, arg178_1, buf231, buf233, 4816896, grid=grid(4816896), stream=stream0)
        buf234 = reinterpret_tensor(buf202, (128, 48, 48), (2304, 48, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [matmul_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf232, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf233, (128, 784, 48), (37632, 48, 1), 0), out=buf234)
        buf237 = reinterpret_tensor(buf199, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [attn_91], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf234, arg179_1, buf237, 6144, 48, grid=grid(6144), stream=stream0)
        del arg179_1
        buf238 = reinterpret_tensor(buf233, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_569], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf227, arg178_1, buf238, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg178_1
        buf239 = reinterpret_tensor(buf232, (128, 48, 784), (37632, 784, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_569], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf237, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf238, (128, 48, 784), (37632, 784, 1), 0), out=buf239)
        buf240 = reinterpret_tensor(buf238, (8, 784, 768), (602112, 768, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [x_571], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf239, buf240, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf241 = reinterpret_tensor(buf239, (6272, 768), (768, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [x_571], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (6272, 768), (768, 1), 0), reinterpret_tensor(arg180_1, (768, 768), (1, 768), 0), out=buf241)
        del arg180_1
        buf245 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [x_571, mul_131, x_573, layer_norm_96], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf222, arg174_1, buf241, arg181_1, arg183_1, arg184_1, buf245, 6272, 768, grid=grid(6272), stream=stream0)
        del arg183_1
        del arg184_1
        # Topologically Sorted Source Nodes: [x_575], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(reinterpret_tensor(buf245, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg185_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf246, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg185_1
        del buf245
        buf247 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [x_575, x_576, x_577], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf247, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg186_1
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        # Topologically Sorted Source Nodes: [x_575, x_576, x_577, x_578], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf248 = extern_kernels.convolution(buf247, arg191_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf248, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg191_1
        buf249 = reinterpret_tensor(buf248, (8, 784, 768), (602112, 768, 1), 0); del buf248  # reuse
        buf253 = reinterpret_tensor(buf247, (8, 784, 768), (602112, 768, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [x_571, mul_131, x_573, mul_132, x_580, layer_norm_97], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf249, buf222, arg174_1, buf241, arg181_1, arg182_1, arg192_1, arg194_1, arg195_1, buf253, 6272, 768, grid=grid(6272), stream=stream0)
        del arg174_1
        del arg181_1
        del arg182_1
        del arg192_1
        del arg194_1
        del arg195_1
        buf254 = reinterpret_tensor(buf220, (6272, 3072), (3072, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (6272, 768), (768, 1), 0), reinterpret_tensor(arg196_1, (768, 3072), (1, 768), 0), out=buf254)
        del arg196_1
        buf255 = reinterpret_tensor(buf254, (8, 784, 3072), (2408448, 3072, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [x_582], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf255, arg197_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg197_1
        buf256 = reinterpret_tensor(buf253, (6272, 768), (768, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf255, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg198_1, (3072, 768), (1, 3072), 0), out=buf256)
        del arg198_1
        buf260 = reinterpret_tensor(buf241, (8, 784, 768), (602112, 768, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [mul_133, x_586, layer_norm_98], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf249, arg193_1, buf256, arg199_1, arg201_1, arg202_1, buf260, 6272, 768, grid=grid(6272), stream=stream0)
        del arg201_1
        del arg202_1
        buf261 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (6272, 768), (768, 1), 0), reinterpret_tensor(arg203_1, (768, 2304), (1, 768), 0), out=buf261)
        del arg203_1
        buf262 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [q_65], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf261, arg204_1, buf262, 43008, 112, grid=grid(43008), stream=stream0)
        buf263 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [q_65], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf262, buf263, 6144, 7, grid=grid(6144), stream=stream0)
        buf264 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [k_65], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf261, arg204_1, buf264, 43008, 112, grid=grid(43008), stream=stream0)
        buf265 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [k_65], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf264, buf265, 6144, 7, grid=grid(6144), stream=stream0)
        buf266 = reinterpret_tensor(buf260, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [q_65, matmul_62], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf261, arg204_1, buf263, buf266, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf267 = reinterpret_tensor(buf222, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [matmul_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf261, arg204_1, buf265, buf267, 4816896, grid=grid(4816896), stream=stream0)
        buf268 = reinterpret_tensor(buf237, (128, 48, 48), (2304, 48, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [matmul_62], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf266, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf267, (128, 784, 48), (37632, 48, 1), 0), out=buf268)
        buf271 = reinterpret_tensor(buf234, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [attn_94], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf268, arg205_1, buf271, 6144, 48, grid=grid(6144), stream=stream0)
        del arg205_1
        buf272 = reinterpret_tensor(buf267, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [x_587], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf261, arg204_1, buf272, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg204_1
        buf273 = reinterpret_tensor(buf266, (128, 48, 784), (37632, 784, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [x_587], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf271, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf272, (128, 48, 784), (37632, 784, 1), 0), out=buf273)
        buf274 = reinterpret_tensor(buf272, (8, 784, 768), (602112, 768, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [x_589], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf273, buf274, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf275 = reinterpret_tensor(buf273, (6272, 768), (768, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [x_589], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (6272, 768), (768, 1), 0), reinterpret_tensor(arg206_1, (768, 768), (1, 768), 0), out=buf275)
        del arg206_1
        buf276 = reinterpret_tensor(buf275, (8, 784, 768), (602112, 768, 1), 0); del buf275  # reuse
        buf280 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [mul_133, x_586, x_589, mul_135, x_591, layer_norm_99], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf276, buf249, arg193_1, buf256, arg199_1, arg200_1, arg207_1, arg209_1, arg210_1, buf280, 6272, 768, grid=grid(6272), stream=stream0)
        del arg193_1
        del arg199_1
        del arg200_1
        del arg207_1
        del arg209_1
        del arg210_1
        del buf249
        del buf256
        # Topologically Sorted Source Nodes: [x_593], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(reinterpret_tensor(buf280, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg211_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf281, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg211_1
        buf282 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [x_593, x_594, x_595], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf282, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        del arg216_1
        # Topologically Sorted Source Nodes: [x_593, x_594, x_595, x_596], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf283 = extern_kernels.convolution(buf282, arg217_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf283, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg217_1
        buf287 = reinterpret_tensor(buf282, (8, 784, 768), (602112, 768, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [mul_136, x_598, layer_norm_100], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf276, arg208_1, buf283, arg218_1, arg220_1, arg221_1, buf287, 6272, 768, grid=grid(6272), stream=stream0)
        del arg220_1
        del arg221_1
        buf288 = reinterpret_tensor(buf255, (6272, 3072), (3072, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf287, (6272, 768), (768, 1), 0), reinterpret_tensor(arg222_1, (768, 3072), (1, 768), 0), out=buf288)
        del arg222_1
        buf289 = reinterpret_tensor(buf288, (8, 784, 3072), (2408448, 3072, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [x_600], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf289, arg223_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg223_1
        buf290 = reinterpret_tensor(buf287, (6272, 768), (768, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf289, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg224_1, (3072, 768), (1, 3072), 0), out=buf290)
        del arg224_1
        buf291 = reinterpret_tensor(buf290, (8, 784, 768), (602112, 768, 1), 0); del buf290  # reuse
        buf295 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [mul_136, x_598, mul_137, x_604, layer_norm_101], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf291, buf276, arg208_1, buf283, arg218_1, arg219_1, arg225_1, arg227_1, arg228_1, buf295, 6272, 768, grid=grid(6272), stream=stream0)
        del arg208_1
        del arg218_1
        del arg219_1
        del arg225_1
        del arg227_1
        del arg228_1
        del buf276
        buf296 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf295, (6272, 768), (768, 1), 0), reinterpret_tensor(arg229_1, (768, 2304), (1, 768), 0), out=buf296)
        del arg229_1
        buf297 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [q_67], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf296, arg230_1, buf297, 43008, 112, grid=grid(43008), stream=stream0)
        buf298 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [q_67], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf297, buf298, 6144, 7, grid=grid(6144), stream=stream0)
        buf299 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [k_67], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf296, arg230_1, buf299, 43008, 112, grid=grid(43008), stream=stream0)
        buf300 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [k_67], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf299, buf300, 6144, 7, grid=grid(6144), stream=stream0)
        buf301 = reinterpret_tensor(buf295, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [q_67, matmul_64], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf296, arg230_1, buf298, buf301, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf302 = reinterpret_tensor(buf283, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [matmul_64], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf296, arg230_1, buf300, buf302, 4816896, grid=grid(4816896), stream=stream0)
        buf303 = reinterpret_tensor(buf271, (128, 48, 48), (2304, 48, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [matmul_64], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf301, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf302, (128, 784, 48), (37632, 48, 1), 0), out=buf303)
        buf306 = reinterpret_tensor(buf268, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [attn_97], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf303, arg231_1, buf306, 6144, 48, grid=grid(6144), stream=stream0)
        del arg231_1
        buf307 = reinterpret_tensor(buf302, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [x_605], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf296, arg230_1, buf307, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg230_1
        buf308 = reinterpret_tensor(buf301, (128, 48, 784), (37632, 784, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [x_605], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf306, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf307, (128, 48, 784), (37632, 784, 1), 0), out=buf308)
        buf309 = reinterpret_tensor(buf307, (8, 784, 768), (602112, 768, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [x_607], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf308, buf309, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf310 = reinterpret_tensor(buf308, (6272, 768), (768, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [x_607], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (6272, 768), (768, 1), 0), reinterpret_tensor(arg232_1, (768, 768), (1, 768), 0), out=buf310)
        del arg232_1
        buf314 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [x_607, mul_139, x_609, layer_norm_102], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf291, arg226_1, buf310, arg233_1, arg235_1, arg236_1, buf314, 6272, 768, grid=grid(6272), stream=stream0)
        del arg235_1
        del arg236_1
        # Topologically Sorted Source Nodes: [x_611], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(reinterpret_tensor(buf314, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg237_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf315, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg237_1
        del buf314
        buf316 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [x_611, x_612, x_613], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf316, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg238_1
        del arg239_1
        del arg240_1
        del arg241_1
        del arg242_1
        # Topologically Sorted Source Nodes: [x_611, x_612, x_613, x_614], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf317 = extern_kernels.convolution(buf316, arg243_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf317, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg243_1
        buf318 = reinterpret_tensor(buf317, (8, 784, 768), (602112, 768, 1), 0); del buf317  # reuse
        buf322 = reinterpret_tensor(buf316, (8, 784, 768), (602112, 768, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [x_607, mul_139, x_609, mul_140, x_616, layer_norm_103], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf318, buf291, arg226_1, buf310, arg233_1, arg234_1, arg244_1, arg246_1, arg247_1, buf322, 6272, 768, grid=grid(6272), stream=stream0)
        del arg226_1
        del arg233_1
        del arg234_1
        del arg244_1
        del arg246_1
        del arg247_1
        buf323 = reinterpret_tensor(buf289, (6272, 3072), (3072, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf322, (6272, 768), (768, 1), 0), reinterpret_tensor(arg248_1, (768, 3072), (1, 768), 0), out=buf323)
        del arg248_1
        buf324 = reinterpret_tensor(buf323, (8, 784, 3072), (2408448, 3072, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [x_618], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf324, arg249_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg249_1
        buf325 = reinterpret_tensor(buf322, (6272, 768), (768, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg250_1, (3072, 768), (1, 3072), 0), out=buf325)
        del arg250_1
        buf329 = reinterpret_tensor(buf310, (8, 784, 768), (602112, 768, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [mul_141, x_622, layer_norm_104], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf318, arg245_1, buf325, arg251_1, arg253_1, arg254_1, buf329, 6272, 768, grid=grid(6272), stream=stream0)
        del arg253_1
        del arg254_1
        buf330 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf329, (6272, 768), (768, 1), 0), reinterpret_tensor(arg255_1, (768, 2304), (1, 768), 0), out=buf330)
        del arg255_1
        buf331 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [q_69], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf330, arg256_1, buf331, 43008, 112, grid=grid(43008), stream=stream0)
        buf332 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [q_69], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf331, buf332, 6144, 7, grid=grid(6144), stream=stream0)
        buf333 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [k_69], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf330, arg256_1, buf333, 43008, 112, grid=grid(43008), stream=stream0)
        buf334 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [k_69], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf333, buf334, 6144, 7, grid=grid(6144), stream=stream0)
        buf335 = reinterpret_tensor(buf329, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [q_69, matmul_66], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf330, arg256_1, buf332, buf335, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf336 = reinterpret_tensor(buf291, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [matmul_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf330, arg256_1, buf334, buf336, 4816896, grid=grid(4816896), stream=stream0)
        buf337 = reinterpret_tensor(buf306, (128, 48, 48), (2304, 48, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [matmul_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf336, (128, 784, 48), (37632, 48, 1), 0), out=buf337)
        buf340 = reinterpret_tensor(buf303, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [attn_100], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf337, arg257_1, buf340, 6144, 48, grid=grid(6144), stream=stream0)
        del arg257_1
        buf341 = reinterpret_tensor(buf336, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [x_623], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf330, arg256_1, buf341, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg256_1
        buf342 = reinterpret_tensor(buf335, (128, 48, 784), (37632, 784, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [x_623], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf340, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf341, (128, 48, 784), (37632, 784, 1), 0), out=buf342)
        buf343 = reinterpret_tensor(buf341, (8, 784, 768), (602112, 768, 1), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [x_625], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf342, buf343, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf344 = reinterpret_tensor(buf342, (6272, 768), (768, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [x_625], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf343, (6272, 768), (768, 1), 0), reinterpret_tensor(arg258_1, (768, 768), (1, 768), 0), out=buf344)
        del arg258_1
        buf345 = reinterpret_tensor(buf344, (8, 784, 768), (602112, 768, 1), 0); del buf344  # reuse
        buf349 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [mul_141, x_622, x_625, mul_143, x_627, layer_norm_105], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf345, buf318, arg245_1, buf325, arg251_1, arg252_1, arg259_1, arg261_1, arg262_1, buf349, 6272, 768, grid=grid(6272), stream=stream0)
        del arg245_1
        del arg251_1
        del arg252_1
        del arg259_1
        del arg261_1
        del arg262_1
        del buf318
        del buf325
        # Topologically Sorted Source Nodes: [x_629], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(reinterpret_tensor(buf349, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg263_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf350, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg263_1
        buf351 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [x_629, x_630, x_631], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf351, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg264_1
        del arg265_1
        del arg266_1
        del arg267_1
        del arg268_1
        # Topologically Sorted Source Nodes: [x_629, x_630, x_631, x_632], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf352 = extern_kernels.convolution(buf351, arg269_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf352, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg269_1
        buf356 = reinterpret_tensor(buf351, (8, 784, 768), (602112, 768, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [mul_144, x_634, layer_norm_106], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf345, arg260_1, buf352, arg270_1, arg272_1, arg273_1, buf356, 6272, 768, grid=grid(6272), stream=stream0)
        del arg272_1
        del arg273_1
        buf357 = reinterpret_tensor(buf324, (6272, 3072), (3072, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (6272, 768), (768, 1), 0), reinterpret_tensor(arg274_1, (768, 3072), (1, 768), 0), out=buf357)
        del arg274_1
        buf358 = reinterpret_tensor(buf357, (8, 784, 3072), (2408448, 3072, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [x_636], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf358, arg275_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg275_1
        buf359 = reinterpret_tensor(buf356, (6272, 768), (768, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg276_1, (3072, 768), (1, 3072), 0), out=buf359)
        del arg276_1
        buf360 = reinterpret_tensor(buf359, (8, 784, 768), (602112, 768, 1), 0); del buf359  # reuse
        buf364 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [mul_144, x_634, mul_145, x_640, layer_norm_107], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf360, buf345, arg260_1, buf352, arg270_1, arg271_1, arg277_1, arg279_1, arg280_1, buf364, 6272, 768, grid=grid(6272), stream=stream0)
        del arg260_1
        del arg270_1
        del arg271_1
        del arg277_1
        del arg279_1
        del arg280_1
        del buf345
        buf365 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (6272, 768), (768, 1), 0), reinterpret_tensor(arg281_1, (768, 2304), (1, 768), 0), out=buf365)
        del arg281_1
        buf366 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [q_71], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf365, arg282_1, buf366, 43008, 112, grid=grid(43008), stream=stream0)
        buf367 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [q_71], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf366, buf367, 6144, 7, grid=grid(6144), stream=stream0)
        buf368 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [k_71], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf365, arg282_1, buf368, 43008, 112, grid=grid(43008), stream=stream0)
        buf369 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [k_71], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf368, buf369, 6144, 7, grid=grid(6144), stream=stream0)
        buf370 = reinterpret_tensor(buf364, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [q_71, matmul_68], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf365, arg282_1, buf367, buf370, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf371 = reinterpret_tensor(buf352, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [matmul_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf365, arg282_1, buf369, buf371, 4816896, grid=grid(4816896), stream=stream0)
        buf372 = reinterpret_tensor(buf340, (128, 48, 48), (2304, 48, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [matmul_68], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf370, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf371, (128, 784, 48), (37632, 48, 1), 0), out=buf372)
        buf375 = reinterpret_tensor(buf337, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [attn_103], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf372, arg283_1, buf375, 6144, 48, grid=grid(6144), stream=stream0)
        del arg283_1
        buf376 = reinterpret_tensor(buf371, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf371  # reuse
        # Topologically Sorted Source Nodes: [x_641], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf365, arg282_1, buf376, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg282_1
        buf377 = reinterpret_tensor(buf370, (128, 48, 784), (37632, 784, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [x_641], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf375, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf376, (128, 48, 784), (37632, 784, 1), 0), out=buf377)
        buf378 = reinterpret_tensor(buf376, (8, 784, 768), (602112, 768, 1), 0); del buf376  # reuse
        # Topologically Sorted Source Nodes: [x_643], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf377, buf378, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf379 = reinterpret_tensor(buf377, (6272, 768), (768, 1), 0); del buf377  # reuse
        # Topologically Sorted Source Nodes: [x_643], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf378, (6272, 768), (768, 1), 0), reinterpret_tensor(arg284_1, (768, 768), (1, 768), 0), out=buf379)
        del arg284_1
        buf383 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [x_643, mul_147, x_645, layer_norm_108], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf360, arg278_1, buf379, arg285_1, arg287_1, arg288_1, buf383, 6272, 768, grid=grid(6272), stream=stream0)
        del arg287_1
        del arg288_1
        # Topologically Sorted Source Nodes: [x_647], Original ATen: [aten.convolution]
        buf384 = extern_kernels.convolution(reinterpret_tensor(buf383, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg289_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf384, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg289_1
        del buf383
        buf385 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [x_647, x_648, x_649], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf385, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg290_1
        del arg291_1
        del arg292_1
        del arg293_1
        del arg294_1
        # Topologically Sorted Source Nodes: [x_647, x_648, x_649, x_650], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf386 = extern_kernels.convolution(buf385, arg295_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf386, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg295_1
        buf387 = reinterpret_tensor(buf386, (8, 784, 768), (602112, 768, 1), 0); del buf386  # reuse
        buf391 = reinterpret_tensor(buf385, (8, 784, 768), (602112, 768, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [x_643, mul_147, x_645, mul_148, x_652, layer_norm_109], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf387, buf360, arg278_1, buf379, arg285_1, arg286_1, arg296_1, arg298_1, arg299_1, buf391, 6272, 768, grid=grid(6272), stream=stream0)
        del arg278_1
        del arg285_1
        del arg286_1
        del arg296_1
        del arg298_1
        del arg299_1
        buf392 = reinterpret_tensor(buf358, (6272, 3072), (3072, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf391, (6272, 768), (768, 1), 0), reinterpret_tensor(arg300_1, (768, 3072), (1, 768), 0), out=buf392)
        del arg300_1
        buf393 = reinterpret_tensor(buf392, (8, 784, 3072), (2408448, 3072, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [x_654], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf393, arg301_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg301_1
        buf394 = reinterpret_tensor(buf391, (6272, 768), (768, 1), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf393, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg302_1, (3072, 768), (1, 3072), 0), out=buf394)
        del arg302_1
        buf398 = reinterpret_tensor(buf379, (8, 784, 768), (602112, 768, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [mul_149, x_658, layer_norm_110], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf387, arg297_1, buf394, arg303_1, arg305_1, arg306_1, buf398, 6272, 768, grid=grid(6272), stream=stream0)
        del arg305_1
        del arg306_1
        buf399 = buf365; del buf365  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf398, (6272, 768), (768, 1), 0), reinterpret_tensor(arg307_1, (768, 2304), (1, 768), 0), out=buf399)
        del arg307_1
        buf400 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [q_73], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf399, arg308_1, buf400, 43008, 112, grid=grid(43008), stream=stream0)
        buf401 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [q_73], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf400, buf401, 6144, 7, grid=grid(6144), stream=stream0)
        buf402 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [k_73], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf399, arg308_1, buf402, 43008, 112, grid=grid(43008), stream=stream0)
        buf403 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [k_73], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf402, buf403, 6144, 7, grid=grid(6144), stream=stream0)
        buf404 = reinterpret_tensor(buf398, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [q_73, matmul_70], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf399, arg308_1, buf401, buf404, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf405 = reinterpret_tensor(buf360, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [matmul_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf399, arg308_1, buf403, buf405, 4816896, grid=grid(4816896), stream=stream0)
        buf406 = reinterpret_tensor(buf375, (128, 48, 48), (2304, 48, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [matmul_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf404, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf405, (128, 784, 48), (37632, 48, 1), 0), out=buf406)
        buf409 = reinterpret_tensor(buf372, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [attn_106], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf406, arg309_1, buf409, 6144, 48, grid=grid(6144), stream=stream0)
        del arg309_1
        buf410 = reinterpret_tensor(buf405, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [x_659], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf399, arg308_1, buf410, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg308_1
        buf411 = reinterpret_tensor(buf404, (128, 48, 784), (37632, 784, 1), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [x_659], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf409, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf410, (128, 48, 784), (37632, 784, 1), 0), out=buf411)
        buf412 = reinterpret_tensor(buf410, (8, 784, 768), (602112, 768, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [x_661], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf411, buf412, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf413 = reinterpret_tensor(buf411, (6272, 768), (768, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [x_661], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (6272, 768), (768, 1), 0), reinterpret_tensor(arg310_1, (768, 768), (1, 768), 0), out=buf413)
        del arg310_1
        buf414 = reinterpret_tensor(buf413, (8, 784, 768), (602112, 768, 1), 0); del buf413  # reuse
        buf418 = buf412; del buf412  # reuse
        # Topologically Sorted Source Nodes: [mul_149, x_658, x_661, mul_151, x_663, layer_norm_111], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf414, buf387, arg297_1, buf394, arg303_1, arg304_1, arg311_1, arg313_1, arg314_1, buf418, 6272, 768, grid=grid(6272), stream=stream0)
        del arg297_1
        del arg303_1
        del arg304_1
        del arg311_1
        del arg313_1
        del arg314_1
        del buf387
        del buf394
        # Topologically Sorted Source Nodes: [x_665], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(reinterpret_tensor(buf418, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg315_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf419, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg315_1
        buf420 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [x_665, x_666, x_667], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf420, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg316_1
        del arg317_1
        del arg318_1
        del arg319_1
        del arg320_1
        # Topologically Sorted Source Nodes: [x_665, x_666, x_667, x_668], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf421 = extern_kernels.convolution(buf420, arg321_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf421, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg321_1
        buf425 = reinterpret_tensor(buf420, (8, 784, 768), (602112, 768, 1), 0); del buf420  # reuse
        # Topologically Sorted Source Nodes: [mul_152, x_670, layer_norm_112], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf414, arg312_1, buf421, arg322_1, arg324_1, arg325_1, buf425, 6272, 768, grid=grid(6272), stream=stream0)
        del arg324_1
        del arg325_1
        buf426 = reinterpret_tensor(buf393, (6272, 3072), (3072, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf425, (6272, 768), (768, 1), 0), reinterpret_tensor(arg326_1, (768, 3072), (1, 768), 0), out=buf426)
        del arg326_1
        buf427 = reinterpret_tensor(buf426, (8, 784, 3072), (2408448, 3072, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [x_672], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf427, arg327_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg327_1
        buf428 = reinterpret_tensor(buf425, (6272, 768), (768, 1), 0); del buf425  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf427, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg328_1, (3072, 768), (1, 3072), 0), out=buf428)
        del arg328_1
        buf429 = reinterpret_tensor(buf428, (8, 784, 768), (602112, 768, 1), 0); del buf428  # reuse
        buf433 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [mul_152, x_670, mul_153, x_676, layer_norm_113], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf429, buf414, arg312_1, buf421, arg322_1, arg323_1, arg329_1, arg331_1, arg332_1, buf433, 6272, 768, grid=grid(6272), stream=stream0)
        del arg312_1
        del arg322_1
        del arg323_1
        del arg329_1
        del arg331_1
        del arg332_1
        del buf414
        buf434 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf433, (6272, 768), (768, 1), 0), reinterpret_tensor(arg333_1, (768, 2304), (1, 768), 0), out=buf434)
        del arg333_1
        buf435 = buf402; del buf402  # reuse
        # Topologically Sorted Source Nodes: [q_75], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf434, arg334_1, buf435, 43008, 112, grid=grid(43008), stream=stream0)
        buf436 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [q_75], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf435, buf436, 6144, 7, grid=grid(6144), stream=stream0)
        buf437 = buf435; del buf435  # reuse
        # Topologically Sorted Source Nodes: [k_75], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf434, arg334_1, buf437, 43008, 112, grid=grid(43008), stream=stream0)
        buf438 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [k_75], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf437, buf438, 6144, 7, grid=grid(6144), stream=stream0)
        buf439 = reinterpret_tensor(buf433, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf433  # reuse
        # Topologically Sorted Source Nodes: [q_75, matmul_72], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf434, arg334_1, buf436, buf439, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf440 = reinterpret_tensor(buf421, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf421  # reuse
        # Topologically Sorted Source Nodes: [matmul_72], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf434, arg334_1, buf438, buf440, 4816896, grid=grid(4816896), stream=stream0)
        buf441 = reinterpret_tensor(buf409, (128, 48, 48), (2304, 48, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [matmul_72], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf439, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf440, (128, 784, 48), (37632, 48, 1), 0), out=buf441)
        buf444 = reinterpret_tensor(buf406, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [attn_109], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf441, arg335_1, buf444, 6144, 48, grid=grid(6144), stream=stream0)
        del arg335_1
        buf445 = reinterpret_tensor(buf440, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [x_677], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf434, arg334_1, buf445, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg334_1
        buf446 = reinterpret_tensor(buf439, (128, 48, 784), (37632, 784, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [x_677], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf444, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf445, (128, 48, 784), (37632, 784, 1), 0), out=buf446)
        buf447 = reinterpret_tensor(buf445, (8, 784, 768), (602112, 768, 1), 0); del buf445  # reuse
        # Topologically Sorted Source Nodes: [x_679], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf446, buf447, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf448 = reinterpret_tensor(buf446, (6272, 768), (768, 1), 0); del buf446  # reuse
        # Topologically Sorted Source Nodes: [x_679], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (6272, 768), (768, 1), 0), reinterpret_tensor(arg336_1, (768, 768), (1, 768), 0), out=buf448)
        del arg336_1
        buf452 = buf447; del buf447  # reuse
        # Topologically Sorted Source Nodes: [x_679, mul_155, x_681, layer_norm_114], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf429, arg330_1, buf448, arg337_1, arg339_1, arg340_1, buf452, 6272, 768, grid=grid(6272), stream=stream0)
        del arg339_1
        del arg340_1
        # Topologically Sorted Source Nodes: [x_683], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(reinterpret_tensor(buf452, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg341_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf453, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg341_1
        del buf452
        buf454 = buf453; del buf453  # reuse
        # Topologically Sorted Source Nodes: [x_683, x_684, x_685], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf454, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        del arg346_1
        # Topologically Sorted Source Nodes: [x_683, x_684, x_685, x_686], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf455 = extern_kernels.convolution(buf454, arg347_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf455, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg347_1
        buf456 = reinterpret_tensor(buf455, (8, 784, 768), (602112, 768, 1), 0); del buf455  # reuse
        buf460 = reinterpret_tensor(buf454, (8, 784, 768), (602112, 768, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [x_679, mul_155, x_681, mul_156, x_688, layer_norm_115], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf456, buf429, arg330_1, buf448, arg337_1, arg338_1, arg348_1, arg350_1, arg351_1, buf460, 6272, 768, grid=grid(6272), stream=stream0)
        del arg330_1
        del arg337_1
        del arg338_1
        del arg348_1
        del arg350_1
        del arg351_1
        buf461 = reinterpret_tensor(buf427, (6272, 3072), (3072, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf460, (6272, 768), (768, 1), 0), reinterpret_tensor(arg352_1, (768, 3072), (1, 768), 0), out=buf461)
        del arg352_1
        buf462 = reinterpret_tensor(buf461, (8, 784, 3072), (2408448, 3072, 1), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [x_690], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf462, arg353_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg353_1
        buf463 = reinterpret_tensor(buf460, (6272, 768), (768, 1), 0); del buf460  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf462, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg354_1, (3072, 768), (1, 3072), 0), out=buf463)
        del arg354_1
        buf467 = reinterpret_tensor(buf448, (8, 784, 768), (602112, 768, 1), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [mul_157, x_694, layer_norm_116], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf456, arg349_1, buf463, arg355_1, arg357_1, arg358_1, buf467, 6272, 768, grid=grid(6272), stream=stream0)
        del arg357_1
        del arg358_1
        buf468 = buf434; del buf434  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf467, (6272, 768), (768, 1), 0), reinterpret_tensor(arg359_1, (768, 2304), (1, 768), 0), out=buf468)
        del arg359_1
        buf469 = buf437; del buf437  # reuse
        # Topologically Sorted Source Nodes: [q_77], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf468, arg360_1, buf469, 43008, 112, grid=grid(43008), stream=stream0)
        buf470 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [q_77], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf469, buf470, 6144, 7, grid=grid(6144), stream=stream0)
        buf471 = buf469; del buf469  # reuse
        # Topologically Sorted Source Nodes: [k_77], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf468, arg360_1, buf471, 43008, 112, grid=grid(43008), stream=stream0)
        buf472 = buf436; del buf436  # reuse
        # Topologically Sorted Source Nodes: [k_77], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf471, buf472, 6144, 7, grid=grid(6144), stream=stream0)
        buf473 = reinterpret_tensor(buf467, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [q_77, matmul_74], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf468, arg360_1, buf470, buf473, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf474 = reinterpret_tensor(buf429, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [matmul_74], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf468, arg360_1, buf472, buf474, 4816896, grid=grid(4816896), stream=stream0)
        buf475 = reinterpret_tensor(buf444, (128, 48, 48), (2304, 48, 1), 0); del buf444  # reuse
        # Topologically Sorted Source Nodes: [matmul_74], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf473, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf474, (128, 784, 48), (37632, 48, 1), 0), out=buf475)
        buf478 = reinterpret_tensor(buf441, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [attn_112], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf475, arg361_1, buf478, 6144, 48, grid=grid(6144), stream=stream0)
        del arg361_1
        buf479 = reinterpret_tensor(buf474, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf474  # reuse
        # Topologically Sorted Source Nodes: [x_695], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf468, arg360_1, buf479, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg360_1
        buf480 = reinterpret_tensor(buf473, (128, 48, 784), (37632, 784, 1), 0); del buf473  # reuse
        # Topologically Sorted Source Nodes: [x_695], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf478, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf479, (128, 48, 784), (37632, 784, 1), 0), out=buf480)
        buf481 = reinterpret_tensor(buf479, (8, 784, 768), (602112, 768, 1), 0); del buf479  # reuse
        # Topologically Sorted Source Nodes: [x_697], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf480, buf481, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf482 = reinterpret_tensor(buf480, (6272, 768), (768, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [x_697], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf481, (6272, 768), (768, 1), 0), reinterpret_tensor(arg362_1, (768, 768), (1, 768), 0), out=buf482)
        del arg362_1
        buf483 = reinterpret_tensor(buf482, (8, 784, 768), (602112, 768, 1), 0); del buf482  # reuse
        buf487 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [mul_157, x_694, x_697, mul_159, x_699, layer_norm_117], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf483, buf456, arg349_1, buf463, arg355_1, arg356_1, arg363_1, arg365_1, arg366_1, buf487, 6272, 768, grid=grid(6272), stream=stream0)
        del arg349_1
        del arg355_1
        del arg356_1
        del arg363_1
        del arg365_1
        del arg366_1
        del buf456
        del buf463
        # Topologically Sorted Source Nodes: [x_701], Original ATen: [aten.convolution]
        buf488 = extern_kernels.convolution(reinterpret_tensor(buf487, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg367_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf488, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg367_1
        buf489 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [x_701, x_702, x_703], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf489, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg368_1
        del arg369_1
        del arg370_1
        del arg371_1
        del arg372_1
        # Topologically Sorted Source Nodes: [x_701, x_702, x_703, x_704], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf490 = extern_kernels.convolution(buf489, arg373_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf490, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg373_1
        buf494 = reinterpret_tensor(buf489, (8, 784, 768), (602112, 768, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [mul_160, x_706, layer_norm_118], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf483, arg364_1, buf490, arg374_1, arg376_1, arg377_1, buf494, 6272, 768, grid=grid(6272), stream=stream0)
        del arg376_1
        del arg377_1
        buf495 = reinterpret_tensor(buf462, (6272, 3072), (3072, 1), 0); del buf462  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf494, (6272, 768), (768, 1), 0), reinterpret_tensor(arg378_1, (768, 3072), (1, 768), 0), out=buf495)
        del arg378_1
        buf496 = reinterpret_tensor(buf495, (8, 784, 3072), (2408448, 3072, 1), 0); del buf495  # reuse
        # Topologically Sorted Source Nodes: [x_708], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf496, arg379_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg379_1
        buf497 = reinterpret_tensor(buf494, (6272, 768), (768, 1), 0); del buf494  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf496, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg380_1, (3072, 768), (1, 3072), 0), out=buf497)
        del arg380_1
        buf498 = reinterpret_tensor(buf497, (8, 784, 768), (602112, 768, 1), 0); del buf497  # reuse
        buf502 = buf487; del buf487  # reuse
        # Topologically Sorted Source Nodes: [mul_160, x_706, mul_161, x_712, layer_norm_119], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf498, buf483, arg364_1, buf490, arg374_1, arg375_1, arg381_1, arg383_1, arg384_1, buf502, 6272, 768, grid=grid(6272), stream=stream0)
        del arg364_1
        del arg374_1
        del arg375_1
        del arg381_1
        del arg383_1
        del arg384_1
        del buf483
        buf503 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf502, (6272, 768), (768, 1), 0), reinterpret_tensor(arg385_1, (768, 2304), (1, 768), 0), out=buf503)
        del arg385_1
        buf504 = buf471; del buf471  # reuse
        # Topologically Sorted Source Nodes: [q_79], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf503, arg386_1, buf504, 43008, 112, grid=grid(43008), stream=stream0)
        buf505 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [q_79], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf504, buf505, 6144, 7, grid=grid(6144), stream=stream0)
        buf506 = buf504; del buf504  # reuse
        # Topologically Sorted Source Nodes: [k_79], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf503, arg386_1, buf506, 43008, 112, grid=grid(43008), stream=stream0)
        buf507 = buf470; del buf470  # reuse
        # Topologically Sorted Source Nodes: [k_79], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf506, buf507, 6144, 7, grid=grid(6144), stream=stream0)
        buf508 = reinterpret_tensor(buf502, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [q_79, matmul_76], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf503, arg386_1, buf505, buf508, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf509 = reinterpret_tensor(buf490, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [matmul_76], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf503, arg386_1, buf507, buf509, 4816896, grid=grid(4816896), stream=stream0)
        buf510 = reinterpret_tensor(buf478, (128, 48, 48), (2304, 48, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [matmul_76], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf508, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf509, (128, 784, 48), (37632, 48, 1), 0), out=buf510)
        buf513 = reinterpret_tensor(buf475, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf475  # reuse
        # Topologically Sorted Source Nodes: [attn_115], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf510, arg387_1, buf513, 6144, 48, grid=grid(6144), stream=stream0)
        del arg387_1
        buf514 = reinterpret_tensor(buf509, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf509  # reuse
        # Topologically Sorted Source Nodes: [x_713], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf503, arg386_1, buf514, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg386_1
        buf515 = reinterpret_tensor(buf508, (128, 48, 784), (37632, 784, 1), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [x_713], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf513, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf514, (128, 48, 784), (37632, 784, 1), 0), out=buf515)
        buf516 = reinterpret_tensor(buf514, (8, 784, 768), (602112, 768, 1), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [x_715], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf515, buf516, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf517 = reinterpret_tensor(buf515, (6272, 768), (768, 1), 0); del buf515  # reuse
        # Topologically Sorted Source Nodes: [x_715], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf516, (6272, 768), (768, 1), 0), reinterpret_tensor(arg388_1, (768, 768), (1, 768), 0), out=buf517)
        del arg388_1
        buf521 = buf516; del buf516  # reuse
        # Topologically Sorted Source Nodes: [x_715, mul_163, x_717, layer_norm_120], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf498, arg382_1, buf517, arg389_1, arg391_1, arg392_1, buf521, 6272, 768, grid=grid(6272), stream=stream0)
        del arg391_1
        del arg392_1
        # Topologically Sorted Source Nodes: [x_719], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(reinterpret_tensor(buf521, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg393_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf522, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg393_1
        del buf521
        buf523 = buf522; del buf522  # reuse
        # Topologically Sorted Source Nodes: [x_719, x_720, x_721], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf523, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg394_1
        del arg395_1
        del arg396_1
        del arg397_1
        del arg398_1
        # Topologically Sorted Source Nodes: [x_719, x_720, x_721, x_722], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf524 = extern_kernels.convolution(buf523, arg399_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf524, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg399_1
        buf525 = reinterpret_tensor(buf524, (8, 784, 768), (602112, 768, 1), 0); del buf524  # reuse
        buf529 = reinterpret_tensor(buf523, (8, 784, 768), (602112, 768, 1), 0); del buf523  # reuse
        # Topologically Sorted Source Nodes: [x_715, mul_163, x_717, mul_164, x_724, layer_norm_121], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf525, buf498, arg382_1, buf517, arg389_1, arg390_1, arg400_1, arg402_1, arg403_1, buf529, 6272, 768, grid=grid(6272), stream=stream0)
        del arg382_1
        del arg389_1
        del arg390_1
        del arg400_1
        del arg402_1
        del arg403_1
        buf530 = reinterpret_tensor(buf496, (6272, 3072), (3072, 1), 0); del buf496  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf529, (6272, 768), (768, 1), 0), reinterpret_tensor(arg404_1, (768, 3072), (1, 768), 0), out=buf530)
        del arg404_1
        buf531 = reinterpret_tensor(buf530, (8, 784, 3072), (2408448, 3072, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [x_726], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf531, arg405_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg405_1
        buf532 = reinterpret_tensor(buf529, (6272, 768), (768, 1), 0); del buf529  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf531, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg406_1, (3072, 768), (1, 3072), 0), out=buf532)
        del arg406_1
        buf536 = reinterpret_tensor(buf517, (8, 784, 768), (602112, 768, 1), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [mul_165, x_730, layer_norm_122], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf525, arg401_1, buf532, arg407_1, arg409_1, arg410_1, buf536, 6272, 768, grid=grid(6272), stream=stream0)
        del arg409_1
        del arg410_1
        buf537 = buf503; del buf503  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf536, (6272, 768), (768, 1), 0), reinterpret_tensor(arg411_1, (768, 2304), (1, 768), 0), out=buf537)
        del arg411_1
        buf538 = buf506; del buf506  # reuse
        # Topologically Sorted Source Nodes: [q_81], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf537, arg412_1, buf538, 43008, 112, grid=grid(43008), stream=stream0)
        buf539 = buf507; del buf507  # reuse
        # Topologically Sorted Source Nodes: [q_81], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf538, buf539, 6144, 7, grid=grid(6144), stream=stream0)
        buf540 = buf538; del buf538  # reuse
        # Topologically Sorted Source Nodes: [k_81], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf537, arg412_1, buf540, 43008, 112, grid=grid(43008), stream=stream0)
        buf541 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [k_81], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf540, buf541, 6144, 7, grid=grid(6144), stream=stream0)
        buf542 = reinterpret_tensor(buf536, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf536  # reuse
        # Topologically Sorted Source Nodes: [q_81, matmul_78], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf537, arg412_1, buf539, buf542, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf543 = reinterpret_tensor(buf498, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf498  # reuse
        # Topologically Sorted Source Nodes: [matmul_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf537, arg412_1, buf541, buf543, 4816896, grid=grid(4816896), stream=stream0)
        buf544 = reinterpret_tensor(buf513, (128, 48, 48), (2304, 48, 1), 0); del buf513  # reuse
        # Topologically Sorted Source Nodes: [matmul_78], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf542, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf543, (128, 784, 48), (37632, 48, 1), 0), out=buf544)
        buf547 = reinterpret_tensor(buf510, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf510  # reuse
        # Topologically Sorted Source Nodes: [attn_118], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf544, arg413_1, buf547, 6144, 48, grid=grid(6144), stream=stream0)
        del arg413_1
        buf548 = reinterpret_tensor(buf543, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [x_731], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf537, arg412_1, buf548, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg412_1
        buf549 = reinterpret_tensor(buf542, (128, 48, 784), (37632, 784, 1), 0); del buf542  # reuse
        # Topologically Sorted Source Nodes: [x_731], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf547, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf548, (128, 48, 784), (37632, 784, 1), 0), out=buf549)
        buf550 = reinterpret_tensor(buf548, (8, 784, 768), (602112, 768, 1), 0); del buf548  # reuse
        # Topologically Sorted Source Nodes: [x_733], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf549, buf550, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf551 = reinterpret_tensor(buf549, (6272, 768), (768, 1), 0); del buf549  # reuse
        # Topologically Sorted Source Nodes: [x_733], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf550, (6272, 768), (768, 1), 0), reinterpret_tensor(arg414_1, (768, 768), (1, 768), 0), out=buf551)
        del arg414_1
        buf552 = reinterpret_tensor(buf551, (8, 784, 768), (602112, 768, 1), 0); del buf551  # reuse
        buf556 = buf550; del buf550  # reuse
        # Topologically Sorted Source Nodes: [mul_165, x_730, x_733, mul_167, x_735, layer_norm_123], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf552, buf525, arg401_1, buf532, arg407_1, arg408_1, arg415_1, arg417_1, arg418_1, buf556, 6272, 768, grid=grid(6272), stream=stream0)
        del arg401_1
        del arg407_1
        del arg408_1
        del arg415_1
        del arg417_1
        del arg418_1
        del buf525
        del buf532
        # Topologically Sorted Source Nodes: [x_737], Original ATen: [aten.convolution]
        buf557 = extern_kernels.convolution(reinterpret_tensor(buf556, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg419_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf557, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg419_1
        buf558 = buf557; del buf557  # reuse
        # Topologically Sorted Source Nodes: [x_737, x_738, x_739], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf558, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg420_1
        del arg421_1
        del arg422_1
        del arg423_1
        del arg424_1
        # Topologically Sorted Source Nodes: [x_737, x_738, x_739, x_740], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf559 = extern_kernels.convolution(buf558, arg425_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf559, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg425_1
        buf563 = reinterpret_tensor(buf558, (8, 784, 768), (602112, 768, 1), 0); del buf558  # reuse
        # Topologically Sorted Source Nodes: [mul_168, x_742, layer_norm_124], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf552, arg416_1, buf559, arg426_1, arg428_1, arg429_1, buf563, 6272, 768, grid=grid(6272), stream=stream0)
        del arg428_1
        del arg429_1
        buf564 = reinterpret_tensor(buf531, (6272, 3072), (3072, 1), 0); del buf531  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf563, (6272, 768), (768, 1), 0), reinterpret_tensor(arg430_1, (768, 3072), (1, 768), 0), out=buf564)
        del arg430_1
        buf565 = reinterpret_tensor(buf564, (8, 784, 3072), (2408448, 3072, 1), 0); del buf564  # reuse
        # Topologically Sorted Source Nodes: [x_744], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf565, arg431_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg431_1
        buf566 = reinterpret_tensor(buf563, (6272, 768), (768, 1), 0); del buf563  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf565, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg432_1, (3072, 768), (1, 3072), 0), out=buf566)
        del arg432_1
        buf567 = reinterpret_tensor(buf566, (8, 784, 768), (602112, 768, 1), 0); del buf566  # reuse
        buf571 = buf556; del buf556  # reuse
        # Topologically Sorted Source Nodes: [mul_168, x_742, mul_169, x_748, layer_norm_125], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf567, buf552, arg416_1, buf559, arg426_1, arg427_1, arg433_1, arg435_1, arg436_1, buf571, 6272, 768, grid=grid(6272), stream=stream0)
        del arg416_1
        del arg426_1
        del arg427_1
        del arg433_1
        del arg435_1
        del arg436_1
        del buf552
        buf572 = buf537; del buf537  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf571, (6272, 768), (768, 1), 0), reinterpret_tensor(arg437_1, (768, 2304), (1, 768), 0), out=buf572)
        del arg437_1
        buf573 = buf540; del buf540  # reuse
        # Topologically Sorted Source Nodes: [q_83], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf572, arg438_1, buf573, 43008, 112, grid=grid(43008), stream=stream0)
        buf574 = buf541; del buf541  # reuse
        # Topologically Sorted Source Nodes: [q_83], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf573, buf574, 6144, 7, grid=grid(6144), stream=stream0)
        buf575 = buf573; del buf573  # reuse
        # Topologically Sorted Source Nodes: [k_83], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf572, arg438_1, buf575, 43008, 112, grid=grid(43008), stream=stream0)
        buf576 = buf539; del buf539  # reuse
        # Topologically Sorted Source Nodes: [k_83], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf575, buf576, 6144, 7, grid=grid(6144), stream=stream0)
        buf577 = reinterpret_tensor(buf571, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [q_83, matmul_80], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf572, arg438_1, buf574, buf577, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf578 = reinterpret_tensor(buf559, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf559  # reuse
        # Topologically Sorted Source Nodes: [matmul_80], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf572, arg438_1, buf576, buf578, 4816896, grid=grid(4816896), stream=stream0)
        buf579 = reinterpret_tensor(buf547, (128, 48, 48), (2304, 48, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [matmul_80], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf577, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf578, (128, 784, 48), (37632, 48, 1), 0), out=buf579)
        buf582 = reinterpret_tensor(buf544, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf544  # reuse
        # Topologically Sorted Source Nodes: [attn_121], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf579, arg439_1, buf582, 6144, 48, grid=grid(6144), stream=stream0)
        del arg439_1
        buf583 = reinterpret_tensor(buf578, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf578  # reuse
        # Topologically Sorted Source Nodes: [x_749], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf572, arg438_1, buf583, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg438_1
        buf584 = reinterpret_tensor(buf577, (128, 48, 784), (37632, 784, 1), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [x_749], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf582, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf583, (128, 48, 784), (37632, 784, 1), 0), out=buf584)
        buf585 = reinterpret_tensor(buf583, (8, 784, 768), (602112, 768, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [x_751], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf584, buf585, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf586 = reinterpret_tensor(buf584, (6272, 768), (768, 1), 0); del buf584  # reuse
        # Topologically Sorted Source Nodes: [x_751], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf585, (6272, 768), (768, 1), 0), reinterpret_tensor(arg440_1, (768, 768), (1, 768), 0), out=buf586)
        del arg440_1
        buf590 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [x_751, mul_171, x_753, layer_norm_126], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf567, arg434_1, buf586, arg441_1, arg443_1, arg444_1, buf590, 6272, 768, grid=grid(6272), stream=stream0)
        del arg443_1
        del arg444_1
        # Topologically Sorted Source Nodes: [x_755], Original ATen: [aten.convolution]
        buf591 = extern_kernels.convolution(reinterpret_tensor(buf590, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg445_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf591, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg445_1
        del buf590
        buf592 = buf591; del buf591  # reuse
        # Topologically Sorted Source Nodes: [x_755, x_756, x_757], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf592, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg446_1
        del arg447_1
        del arg448_1
        del arg449_1
        del arg450_1
        # Topologically Sorted Source Nodes: [x_755, x_756, x_757, x_758], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf593 = extern_kernels.convolution(buf592, arg451_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf593, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg451_1
        buf594 = reinterpret_tensor(buf593, (8, 784, 768), (602112, 768, 1), 0); del buf593  # reuse
        buf598 = reinterpret_tensor(buf592, (8, 784, 768), (602112, 768, 1), 0); del buf592  # reuse
        # Topologically Sorted Source Nodes: [x_751, mul_171, x_753, mul_172, x_760, layer_norm_127], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf594, buf567, arg434_1, buf586, arg441_1, arg442_1, arg452_1, arg454_1, arg455_1, buf598, 6272, 768, grid=grid(6272), stream=stream0)
        del arg434_1
        del arg441_1
        del arg442_1
        del arg452_1
        del arg454_1
        del arg455_1
        buf599 = reinterpret_tensor(buf565, (6272, 3072), (3072, 1), 0); del buf565  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf598, (6272, 768), (768, 1), 0), reinterpret_tensor(arg456_1, (768, 3072), (1, 768), 0), out=buf599)
        del arg456_1
        buf600 = reinterpret_tensor(buf599, (8, 784, 3072), (2408448, 3072, 1), 0); del buf599  # reuse
        # Topologically Sorted Source Nodes: [x_762], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf600, arg457_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg457_1
        buf601 = reinterpret_tensor(buf598, (6272, 768), (768, 1), 0); del buf598  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf600, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg458_1, (3072, 768), (1, 3072), 0), out=buf601)
        del arg458_1
        buf605 = reinterpret_tensor(buf586, (8, 784, 768), (602112, 768, 1), 0); del buf586  # reuse
        # Topologically Sorted Source Nodes: [mul_173, x_766, layer_norm_128], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf594, arg453_1, buf601, arg459_1, arg461_1, arg462_1, buf605, 6272, 768, grid=grid(6272), stream=stream0)
        del arg461_1
        del arg462_1
        buf606 = buf572; del buf572  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf605, (6272, 768), (768, 1), 0), reinterpret_tensor(arg463_1, (768, 2304), (1, 768), 0), out=buf606)
        del arg463_1
        buf607 = buf575; del buf575  # reuse
        # Topologically Sorted Source Nodes: [q_85], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf606, arg464_1, buf607, 43008, 112, grid=grid(43008), stream=stream0)
        buf608 = buf576; del buf576  # reuse
        # Topologically Sorted Source Nodes: [q_85], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf607, buf608, 6144, 7, grid=grid(6144), stream=stream0)
        buf609 = buf607; del buf607  # reuse
        # Topologically Sorted Source Nodes: [k_85], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf606, arg464_1, buf609, 43008, 112, grid=grid(43008), stream=stream0)
        buf610 = buf574; del buf574  # reuse
        # Topologically Sorted Source Nodes: [k_85], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf609, buf610, 6144, 7, grid=grid(6144), stream=stream0)
        buf611 = reinterpret_tensor(buf605, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [q_85, matmul_82], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf606, arg464_1, buf608, buf611, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf612 = reinterpret_tensor(buf567, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf567  # reuse
        # Topologically Sorted Source Nodes: [matmul_82], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf606, arg464_1, buf610, buf612, 4816896, grid=grid(4816896), stream=stream0)
        buf613 = reinterpret_tensor(buf582, (128, 48, 48), (2304, 48, 1), 0); del buf582  # reuse
        # Topologically Sorted Source Nodes: [matmul_82], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf611, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf612, (128, 784, 48), (37632, 48, 1), 0), out=buf613)
        buf616 = reinterpret_tensor(buf579, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf579  # reuse
        # Topologically Sorted Source Nodes: [attn_124], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf613, arg465_1, buf616, 6144, 48, grid=grid(6144), stream=stream0)
        del arg465_1
        buf617 = reinterpret_tensor(buf612, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf612  # reuse
        # Topologically Sorted Source Nodes: [x_767], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf606, arg464_1, buf617, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg464_1
        buf618 = reinterpret_tensor(buf611, (128, 48, 784), (37632, 784, 1), 0); del buf611  # reuse
        # Topologically Sorted Source Nodes: [x_767], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf616, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf617, (128, 48, 784), (37632, 784, 1), 0), out=buf618)
        buf619 = reinterpret_tensor(buf617, (8, 784, 768), (602112, 768, 1), 0); del buf617  # reuse
        # Topologically Sorted Source Nodes: [x_769], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf618, buf619, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf620 = reinterpret_tensor(buf618, (6272, 768), (768, 1), 0); del buf618  # reuse
        # Topologically Sorted Source Nodes: [x_769], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf619, (6272, 768), (768, 1), 0), reinterpret_tensor(arg466_1, (768, 768), (1, 768), 0), out=buf620)
        del arg466_1
        buf621 = reinterpret_tensor(buf620, (8, 784, 768), (602112, 768, 1), 0); del buf620  # reuse
        buf625 = buf619; del buf619  # reuse
        # Topologically Sorted Source Nodes: [mul_173, x_766, x_769, mul_175, x_771, layer_norm_129], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf621, buf594, arg453_1, buf601, arg459_1, arg460_1, arg467_1, arg469_1, arg470_1, buf625, 6272, 768, grid=grid(6272), stream=stream0)
        del arg453_1
        del arg459_1
        del arg460_1
        del arg467_1
        del arg469_1
        del arg470_1
        del buf594
        del buf601
        # Topologically Sorted Source Nodes: [x_773], Original ATen: [aten.convolution]
        buf626 = extern_kernels.convolution(reinterpret_tensor(buf625, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg471_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf626, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg471_1
        buf627 = buf626; del buf626  # reuse
        # Topologically Sorted Source Nodes: [x_773, x_774, x_775], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf627, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg472_1
        del arg473_1
        del arg474_1
        del arg475_1
        del arg476_1
        # Topologically Sorted Source Nodes: [x_773, x_774, x_775, x_776], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf628 = extern_kernels.convolution(buf627, arg477_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf628, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg477_1
        buf632 = reinterpret_tensor(buf627, (8, 784, 768), (602112, 768, 1), 0); del buf627  # reuse
        # Topologically Sorted Source Nodes: [mul_176, x_778, layer_norm_130], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf621, arg468_1, buf628, arg478_1, arg480_1, arg481_1, buf632, 6272, 768, grid=grid(6272), stream=stream0)
        del arg480_1
        del arg481_1
        buf633 = reinterpret_tensor(buf600, (6272, 3072), (3072, 1), 0); del buf600  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf632, (6272, 768), (768, 1), 0), reinterpret_tensor(arg482_1, (768, 3072), (1, 768), 0), out=buf633)
        del arg482_1
        buf634 = reinterpret_tensor(buf633, (8, 784, 3072), (2408448, 3072, 1), 0); del buf633  # reuse
        # Topologically Sorted Source Nodes: [x_780], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf634, arg483_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg483_1
        buf635 = reinterpret_tensor(buf632, (6272, 768), (768, 1), 0); del buf632  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf634, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg484_1, (3072, 768), (1, 3072), 0), out=buf635)
        del arg484_1
        buf636 = reinterpret_tensor(buf635, (8, 784, 768), (602112, 768, 1), 0); del buf635  # reuse
        buf640 = buf625; del buf625  # reuse
        # Topologically Sorted Source Nodes: [mul_176, x_778, mul_177, x_784, layer_norm_131], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf636, buf621, arg468_1, buf628, arg478_1, arg479_1, arg485_1, arg487_1, arg488_1, buf640, 6272, 768, grid=grid(6272), stream=stream0)
        del arg468_1
        del arg478_1
        del arg479_1
        del arg485_1
        del arg487_1
        del arg488_1
        del buf621
        buf641 = buf606; del buf606  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf640, (6272, 768), (768, 1), 0), reinterpret_tensor(arg489_1, (768, 2304), (1, 768), 0), out=buf641)
        del arg489_1
        buf642 = buf609; del buf609  # reuse
        # Topologically Sorted Source Nodes: [q_87], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf641, arg490_1, buf642, 43008, 112, grid=grid(43008), stream=stream0)
        buf643 = buf610; del buf610  # reuse
        # Topologically Sorted Source Nodes: [q_87], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf642, buf643, 6144, 7, grid=grid(6144), stream=stream0)
        buf644 = buf642; del buf642  # reuse
        # Topologically Sorted Source Nodes: [k_87], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf641, arg490_1, buf644, 43008, 112, grid=grid(43008), stream=stream0)
        buf645 = buf608; del buf608  # reuse
        # Topologically Sorted Source Nodes: [k_87], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf644, buf645, 6144, 7, grid=grid(6144), stream=stream0)
        buf646 = reinterpret_tensor(buf640, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf640  # reuse
        # Topologically Sorted Source Nodes: [q_87, matmul_84], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf641, arg490_1, buf643, buf646, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf647 = reinterpret_tensor(buf628, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf628  # reuse
        # Topologically Sorted Source Nodes: [matmul_84], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf641, arg490_1, buf645, buf647, 4816896, grid=grid(4816896), stream=stream0)
        buf648 = reinterpret_tensor(buf616, (128, 48, 48), (2304, 48, 1), 0); del buf616  # reuse
        # Topologically Sorted Source Nodes: [matmul_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf646, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf647, (128, 784, 48), (37632, 48, 1), 0), out=buf648)
        buf651 = reinterpret_tensor(buf613, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [attn_127], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf648, arg491_1, buf651, 6144, 48, grid=grid(6144), stream=stream0)
        del arg491_1
        buf652 = reinterpret_tensor(buf647, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf647  # reuse
        # Topologically Sorted Source Nodes: [x_785], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf641, arg490_1, buf652, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg490_1
        buf653 = reinterpret_tensor(buf646, (128, 48, 784), (37632, 784, 1), 0); del buf646  # reuse
        # Topologically Sorted Source Nodes: [x_785], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf651, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf652, (128, 48, 784), (37632, 784, 1), 0), out=buf653)
        buf654 = reinterpret_tensor(buf652, (8, 784, 768), (602112, 768, 1), 0); del buf652  # reuse
        # Topologically Sorted Source Nodes: [x_787], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf653, buf654, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf655 = reinterpret_tensor(buf653, (6272, 768), (768, 1), 0); del buf653  # reuse
        # Topologically Sorted Source Nodes: [x_787], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf654, (6272, 768), (768, 1), 0), reinterpret_tensor(arg492_1, (768, 768), (1, 768), 0), out=buf655)
        del arg492_1
        buf659 = buf654; del buf654  # reuse
        # Topologically Sorted Source Nodes: [x_787, mul_179, x_789, layer_norm_132], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf636, arg486_1, buf655, arg493_1, arg495_1, arg496_1, buf659, 6272, 768, grid=grid(6272), stream=stream0)
        del arg495_1
        del arg496_1
        # Topologically Sorted Source Nodes: [x_791], Original ATen: [aten.convolution]
        buf660 = extern_kernels.convolution(reinterpret_tensor(buf659, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg497_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf660, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg497_1
        del buf659
        buf661 = buf660; del buf660  # reuse
        # Topologically Sorted Source Nodes: [x_791, x_792, x_793], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf661, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg498_1
        del arg499_1
        del arg500_1
        del arg501_1
        del arg502_1
        # Topologically Sorted Source Nodes: [x_791, x_792, x_793, x_794], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf662 = extern_kernels.convolution(buf661, arg503_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf662, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg503_1
        buf663 = reinterpret_tensor(buf662, (8, 784, 768), (602112, 768, 1), 0); del buf662  # reuse
        buf667 = reinterpret_tensor(buf661, (8, 784, 768), (602112, 768, 1), 0); del buf661  # reuse
        # Topologically Sorted Source Nodes: [x_787, mul_179, x_789, mul_180, x_796, layer_norm_133], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf663, buf636, arg486_1, buf655, arg493_1, arg494_1, arg504_1, arg506_1, arg507_1, buf667, 6272, 768, grid=grid(6272), stream=stream0)
        del arg486_1
        del arg493_1
        del arg494_1
        del arg504_1
        del arg506_1
        del arg507_1
        buf668 = reinterpret_tensor(buf634, (6272, 3072), (3072, 1), 0); del buf634  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf667, (6272, 768), (768, 1), 0), reinterpret_tensor(arg508_1, (768, 3072), (1, 768), 0), out=buf668)
        del arg508_1
        buf669 = reinterpret_tensor(buf668, (8, 784, 3072), (2408448, 3072, 1), 0); del buf668  # reuse
        # Topologically Sorted Source Nodes: [x_798], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf669, arg509_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg509_1
        buf670 = reinterpret_tensor(buf667, (6272, 768), (768, 1), 0); del buf667  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf669, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg510_1, (3072, 768), (1, 3072), 0), out=buf670)
        del arg510_1
        buf674 = reinterpret_tensor(buf655, (8, 784, 768), (602112, 768, 1), 0); del buf655  # reuse
        # Topologically Sorted Source Nodes: [mul_181, x_802, layer_norm_134], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf663, arg505_1, buf670, arg511_1, arg513_1, arg514_1, buf674, 6272, 768, grid=grid(6272), stream=stream0)
        del arg513_1
        del arg514_1
        buf675 = buf641; del buf641  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf674, (6272, 768), (768, 1), 0), reinterpret_tensor(arg515_1, (768, 2304), (1, 768), 0), out=buf675)
        del arg515_1
        buf676 = buf644; del buf644  # reuse
        # Topologically Sorted Source Nodes: [q_89], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf675, arg516_1, buf676, 43008, 112, grid=grid(43008), stream=stream0)
        buf677 = buf645; del buf645  # reuse
        # Topologically Sorted Source Nodes: [q_89], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf676, buf677, 6144, 7, grid=grid(6144), stream=stream0)
        buf678 = buf676; del buf676  # reuse
        # Topologically Sorted Source Nodes: [k_89], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf675, arg516_1, buf678, 43008, 112, grid=grid(43008), stream=stream0)
        buf679 = buf643; del buf643  # reuse
        # Topologically Sorted Source Nodes: [k_89], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf678, buf679, 6144, 7, grid=grid(6144), stream=stream0)
        buf680 = reinterpret_tensor(buf674, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf674  # reuse
        # Topologically Sorted Source Nodes: [q_89, matmul_86], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf675, arg516_1, buf677, buf680, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf681 = reinterpret_tensor(buf636, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf636  # reuse
        # Topologically Sorted Source Nodes: [matmul_86], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf675, arg516_1, buf679, buf681, 4816896, grid=grid(4816896), stream=stream0)
        buf682 = reinterpret_tensor(buf651, (128, 48, 48), (2304, 48, 1), 0); del buf651  # reuse
        # Topologically Sorted Source Nodes: [matmul_86], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf680, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf681, (128, 784, 48), (37632, 48, 1), 0), out=buf682)
        buf685 = reinterpret_tensor(buf648, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf648  # reuse
        # Topologically Sorted Source Nodes: [attn_130], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf682, arg517_1, buf685, 6144, 48, grid=grid(6144), stream=stream0)
        del arg517_1
        buf686 = reinterpret_tensor(buf681, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf681  # reuse
        # Topologically Sorted Source Nodes: [x_803], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf675, arg516_1, buf686, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg516_1
        buf687 = reinterpret_tensor(buf680, (128, 48, 784), (37632, 784, 1), 0); del buf680  # reuse
        # Topologically Sorted Source Nodes: [x_803], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf685, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf686, (128, 48, 784), (37632, 784, 1), 0), out=buf687)
        buf688 = reinterpret_tensor(buf686, (8, 784, 768), (602112, 768, 1), 0); del buf686  # reuse
        # Topologically Sorted Source Nodes: [x_805], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf687, buf688, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf689 = reinterpret_tensor(buf687, (6272, 768), (768, 1), 0); del buf687  # reuse
        # Topologically Sorted Source Nodes: [x_805], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf688, (6272, 768), (768, 1), 0), reinterpret_tensor(arg518_1, (768, 768), (1, 768), 0), out=buf689)
        del arg518_1
        buf690 = reinterpret_tensor(buf689, (8, 784, 768), (602112, 768, 1), 0); del buf689  # reuse
        buf694 = buf688; del buf688  # reuse
        # Topologically Sorted Source Nodes: [mul_181, x_802, x_805, mul_183, x_807, layer_norm_135], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf690, buf663, arg505_1, buf670, arg511_1, arg512_1, arg519_1, arg521_1, arg522_1, buf694, 6272, 768, grid=grid(6272), stream=stream0)
        del arg505_1
        del arg511_1
        del arg512_1
        del arg519_1
        del arg521_1
        del arg522_1
        del buf663
        del buf670
        # Topologically Sorted Source Nodes: [x_809], Original ATen: [aten.convolution]
        buf695 = extern_kernels.convolution(reinterpret_tensor(buf694, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg523_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf695, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg523_1
        buf696 = buf695; del buf695  # reuse
        # Topologically Sorted Source Nodes: [x_809, x_810, x_811], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf696, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg524_1
        del arg525_1
        del arg526_1
        del arg527_1
        del arg528_1
        # Topologically Sorted Source Nodes: [x_809, x_810, x_811, x_812], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf697 = extern_kernels.convolution(buf696, arg529_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf697, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg529_1
        buf701 = reinterpret_tensor(buf696, (8, 784, 768), (602112, 768, 1), 0); del buf696  # reuse
        # Topologically Sorted Source Nodes: [mul_184, x_814, layer_norm_136], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf690, arg520_1, buf697, arg530_1, arg532_1, arg533_1, buf701, 6272, 768, grid=grid(6272), stream=stream0)
        del arg532_1
        del arg533_1
        buf702 = reinterpret_tensor(buf669, (6272, 3072), (3072, 1), 0); del buf669  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf701, (6272, 768), (768, 1), 0), reinterpret_tensor(arg534_1, (768, 3072), (1, 768), 0), out=buf702)
        del arg534_1
        buf703 = reinterpret_tensor(buf702, (8, 784, 3072), (2408448, 3072, 1), 0); del buf702  # reuse
        # Topologically Sorted Source Nodes: [x_816], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf703, arg535_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg535_1
        buf704 = reinterpret_tensor(buf701, (6272, 768), (768, 1), 0); del buf701  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf703, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg536_1, (3072, 768), (1, 3072), 0), out=buf704)
        del arg536_1
        buf705 = reinterpret_tensor(buf704, (8, 784, 768), (602112, 768, 1), 0); del buf704  # reuse
        buf709 = buf694; del buf694  # reuse
        # Topologically Sorted Source Nodes: [mul_184, x_814, mul_185, x_820, layer_norm_137], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf705, buf690, arg520_1, buf697, arg530_1, arg531_1, arg537_1, arg539_1, arg540_1, buf709, 6272, 768, grid=grid(6272), stream=stream0)
        del arg520_1
        del arg530_1
        del arg531_1
        del arg537_1
        del arg539_1
        del arg540_1
        del buf690
        buf710 = buf675; del buf675  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf709, (6272, 768), (768, 1), 0), reinterpret_tensor(arg541_1, (768, 2304), (1, 768), 0), out=buf710)
        del arg541_1
        buf711 = buf678; del buf678  # reuse
        # Topologically Sorted Source Nodes: [q_91], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf710, arg542_1, buf711, 43008, 112, grid=grid(43008), stream=stream0)
        buf712 = buf679; del buf679  # reuse
        # Topologically Sorted Source Nodes: [q_91], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf711, buf712, 6144, 7, grid=grid(6144), stream=stream0)
        buf713 = buf711; del buf711  # reuse
        # Topologically Sorted Source Nodes: [k_91], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf710, arg542_1, buf713, 43008, 112, grid=grid(43008), stream=stream0)
        buf714 = buf677; del buf677  # reuse
        # Topologically Sorted Source Nodes: [k_91], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf713, buf714, 6144, 7, grid=grid(6144), stream=stream0)
        buf715 = reinterpret_tensor(buf709, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf709  # reuse
        # Topologically Sorted Source Nodes: [q_91, matmul_88], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf710, arg542_1, buf712, buf715, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf716 = reinterpret_tensor(buf697, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf697  # reuse
        # Topologically Sorted Source Nodes: [matmul_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf710, arg542_1, buf714, buf716, 4816896, grid=grid(4816896), stream=stream0)
        buf717 = reinterpret_tensor(buf685, (128, 48, 48), (2304, 48, 1), 0); del buf685  # reuse
        # Topologically Sorted Source Nodes: [matmul_88], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf715, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf716, (128, 784, 48), (37632, 48, 1), 0), out=buf717)
        buf720 = reinterpret_tensor(buf682, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf682  # reuse
        # Topologically Sorted Source Nodes: [attn_133], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf717, arg543_1, buf720, 6144, 48, grid=grid(6144), stream=stream0)
        del arg543_1
        buf721 = reinterpret_tensor(buf716, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf716  # reuse
        # Topologically Sorted Source Nodes: [x_821], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf710, arg542_1, buf721, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg542_1
        buf722 = reinterpret_tensor(buf715, (128, 48, 784), (37632, 784, 1), 0); del buf715  # reuse
        # Topologically Sorted Source Nodes: [x_821], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf720, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf721, (128, 48, 784), (37632, 784, 1), 0), out=buf722)
        buf723 = reinterpret_tensor(buf721, (8, 784, 768), (602112, 768, 1), 0); del buf721  # reuse
        # Topologically Sorted Source Nodes: [x_823], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf722, buf723, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf724 = reinterpret_tensor(buf722, (6272, 768), (768, 1), 0); del buf722  # reuse
        # Topologically Sorted Source Nodes: [x_823], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf723, (6272, 768), (768, 1), 0), reinterpret_tensor(arg544_1, (768, 768), (1, 768), 0), out=buf724)
        del arg544_1
        buf728 = buf723; del buf723  # reuse
        # Topologically Sorted Source Nodes: [x_823, mul_187, x_825, layer_norm_138], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf705, arg538_1, buf724, arg545_1, arg547_1, arg548_1, buf728, 6272, 768, grid=grid(6272), stream=stream0)
        del arg547_1
        del arg548_1
        # Topologically Sorted Source Nodes: [x_827], Original ATen: [aten.convolution]
        buf729 = extern_kernels.convolution(reinterpret_tensor(buf728, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg549_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf729, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg549_1
        del buf728
        buf730 = buf729; del buf729  # reuse
        # Topologically Sorted Source Nodes: [x_827, x_828, x_829], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf730, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg550_1
        del arg551_1
        del arg552_1
        del arg553_1
        del arg554_1
        # Topologically Sorted Source Nodes: [x_827, x_828, x_829, x_830], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf731 = extern_kernels.convolution(buf730, arg555_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf731, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg555_1
        buf732 = reinterpret_tensor(buf731, (8, 784, 768), (602112, 768, 1), 0); del buf731  # reuse
        buf736 = reinterpret_tensor(buf730, (8, 784, 768), (602112, 768, 1), 0); del buf730  # reuse
        # Topologically Sorted Source Nodes: [x_823, mul_187, x_825, mul_188, x_832, layer_norm_139], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf732, buf705, arg538_1, buf724, arg545_1, arg546_1, arg556_1, arg558_1, arg559_1, buf736, 6272, 768, grid=grid(6272), stream=stream0)
        del arg538_1
        del arg545_1
        del arg546_1
        del arg556_1
        del arg558_1
        del arg559_1
        buf737 = reinterpret_tensor(buf703, (6272, 3072), (3072, 1), 0); del buf703  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf736, (6272, 768), (768, 1), 0), reinterpret_tensor(arg560_1, (768, 3072), (1, 768), 0), out=buf737)
        del arg560_1
        buf738 = reinterpret_tensor(buf737, (8, 784, 3072), (2408448, 3072, 1), 0); del buf737  # reuse
        # Topologically Sorted Source Nodes: [x_834], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf738, arg561_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg561_1
        buf739 = reinterpret_tensor(buf736, (6272, 768), (768, 1), 0); del buf736  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf738, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg562_1, (3072, 768), (1, 3072), 0), out=buf739)
        del arg562_1
        buf743 = reinterpret_tensor(buf724, (8, 784, 768), (602112, 768, 1), 0); del buf724  # reuse
        # Topologically Sorted Source Nodes: [mul_189, x_838, layer_norm_140], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf732, arg557_1, buf739, arg563_1, arg565_1, arg566_1, buf743, 6272, 768, grid=grid(6272), stream=stream0)
        del arg565_1
        del arg566_1
        buf744 = buf710; del buf710  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf743, (6272, 768), (768, 1), 0), reinterpret_tensor(arg567_1, (768, 2304), (1, 768), 0), out=buf744)
        del arg567_1
        buf745 = buf713; del buf713  # reuse
        # Topologically Sorted Source Nodes: [q_93], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf744, arg568_1, buf745, 43008, 112, grid=grid(43008), stream=stream0)
        buf746 = buf714; del buf714  # reuse
        # Topologically Sorted Source Nodes: [q_93], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf745, buf746, 6144, 7, grid=grid(6144), stream=stream0)
        buf747 = buf745; del buf745  # reuse
        # Topologically Sorted Source Nodes: [k_93], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf744, arg568_1, buf747, 43008, 112, grid=grid(43008), stream=stream0)
        buf748 = buf712; del buf712  # reuse
        # Topologically Sorted Source Nodes: [k_93], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf747, buf748, 6144, 7, grid=grid(6144), stream=stream0)
        buf749 = reinterpret_tensor(buf743, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf743  # reuse
        # Topologically Sorted Source Nodes: [q_93, matmul_90], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf744, arg568_1, buf746, buf749, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf750 = reinterpret_tensor(buf705, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf705  # reuse
        # Topologically Sorted Source Nodes: [matmul_90], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf744, arg568_1, buf748, buf750, 4816896, grid=grid(4816896), stream=stream0)
        buf751 = reinterpret_tensor(buf720, (128, 48, 48), (2304, 48, 1), 0); del buf720  # reuse
        # Topologically Sorted Source Nodes: [matmul_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf749, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf750, (128, 784, 48), (37632, 48, 1), 0), out=buf751)
        buf754 = reinterpret_tensor(buf717, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf717  # reuse
        # Topologically Sorted Source Nodes: [attn_136], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf751, arg569_1, buf754, 6144, 48, grid=grid(6144), stream=stream0)
        del arg569_1
        buf755 = reinterpret_tensor(buf750, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf750  # reuse
        # Topologically Sorted Source Nodes: [x_839], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf744, arg568_1, buf755, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg568_1
        buf756 = reinterpret_tensor(buf749, (128, 48, 784), (37632, 784, 1), 0); del buf749  # reuse
        # Topologically Sorted Source Nodes: [x_839], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf754, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf755, (128, 48, 784), (37632, 784, 1), 0), out=buf756)
        buf757 = reinterpret_tensor(buf755, (8, 784, 768), (602112, 768, 1), 0); del buf755  # reuse
        # Topologically Sorted Source Nodes: [x_841], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf756, buf757, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf758 = reinterpret_tensor(buf756, (6272, 768), (768, 1), 0); del buf756  # reuse
        # Topologically Sorted Source Nodes: [x_841], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf757, (6272, 768), (768, 1), 0), reinterpret_tensor(arg570_1, (768, 768), (1, 768), 0), out=buf758)
        del arg570_1
        buf759 = reinterpret_tensor(buf758, (8, 784, 768), (602112, 768, 1), 0); del buf758  # reuse
        buf763 = buf757; del buf757  # reuse
        # Topologically Sorted Source Nodes: [mul_189, x_838, x_841, mul_191, x_843, layer_norm_141], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf759, buf732, arg557_1, buf739, arg563_1, arg564_1, arg571_1, arg573_1, arg574_1, buf763, 6272, 768, grid=grid(6272), stream=stream0)
        del arg557_1
        del arg563_1
        del arg564_1
        del arg571_1
        del arg573_1
        del arg574_1
        del buf732
        del buf739
        # Topologically Sorted Source Nodes: [x_845], Original ATen: [aten.convolution]
        buf764 = extern_kernels.convolution(reinterpret_tensor(buf763, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg575_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf764, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg575_1
        buf765 = buf764; del buf764  # reuse
        # Topologically Sorted Source Nodes: [x_845, x_846, x_847], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf765, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg576_1
        del arg577_1
        del arg578_1
        del arg579_1
        del arg580_1
        # Topologically Sorted Source Nodes: [x_845, x_846, x_847, x_848], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf766 = extern_kernels.convolution(buf765, arg581_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf766, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg581_1
        buf770 = reinterpret_tensor(buf765, (8, 784, 768), (602112, 768, 1), 0); del buf765  # reuse
        # Topologically Sorted Source Nodes: [mul_192, x_850, layer_norm_142], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf759, arg572_1, buf766, arg582_1, arg584_1, arg585_1, buf770, 6272, 768, grid=grid(6272), stream=stream0)
        del arg584_1
        del arg585_1
        buf771 = reinterpret_tensor(buf738, (6272, 3072), (3072, 1), 0); del buf738  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf770, (6272, 768), (768, 1), 0), reinterpret_tensor(arg586_1, (768, 3072), (1, 768), 0), out=buf771)
        del arg586_1
        buf772 = reinterpret_tensor(buf771, (8, 784, 3072), (2408448, 3072, 1), 0); del buf771  # reuse
        # Topologically Sorted Source Nodes: [x_852], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf772, arg587_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg587_1
        buf773 = reinterpret_tensor(buf770, (6272, 768), (768, 1), 0); del buf770  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf772, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg588_1, (3072, 768), (1, 3072), 0), out=buf773)
        del arg588_1
        buf774 = reinterpret_tensor(buf773, (8, 784, 768), (602112, 768, 1), 0); del buf773  # reuse
        buf778 = buf763; del buf763  # reuse
        # Topologically Sorted Source Nodes: [mul_192, x_850, mul_193, x_856, layer_norm_143], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf774, buf759, arg572_1, buf766, arg582_1, arg583_1, arg589_1, arg591_1, arg592_1, buf778, 6272, 768, grid=grid(6272), stream=stream0)
        del arg572_1
        del arg582_1
        del arg583_1
        del arg589_1
        del arg591_1
        del arg592_1
        del buf759
        buf779 = buf744; del buf744  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf778, (6272, 768), (768, 1), 0), reinterpret_tensor(arg593_1, (768, 2304), (1, 768), 0), out=buf779)
        del arg593_1
        buf780 = buf747; del buf747  # reuse
        # Topologically Sorted Source Nodes: [q_95], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf779, arg594_1, buf780, 43008, 112, grid=grid(43008), stream=stream0)
        buf781 = buf748; del buf748  # reuse
        # Topologically Sorted Source Nodes: [q_95], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf780, buf781, 6144, 7, grid=grid(6144), stream=stream0)
        buf782 = buf780; del buf780  # reuse
        # Topologically Sorted Source Nodes: [k_95], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf779, arg594_1, buf782, 43008, 112, grid=grid(43008), stream=stream0)
        buf783 = buf746; del buf746  # reuse
        # Topologically Sorted Source Nodes: [k_95], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf782, buf783, 6144, 7, grid=grid(6144), stream=stream0)
        buf784 = reinterpret_tensor(buf778, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf778  # reuse
        # Topologically Sorted Source Nodes: [q_95, matmul_92], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf779, arg594_1, buf781, buf784, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf785 = reinterpret_tensor(buf766, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf766  # reuse
        # Topologically Sorted Source Nodes: [matmul_92], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf779, arg594_1, buf783, buf785, 4816896, grid=grid(4816896), stream=stream0)
        buf786 = reinterpret_tensor(buf754, (128, 48, 48), (2304, 48, 1), 0); del buf754  # reuse
        # Topologically Sorted Source Nodes: [matmul_92], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf784, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf785, (128, 784, 48), (37632, 48, 1), 0), out=buf786)
        buf789 = reinterpret_tensor(buf751, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf751  # reuse
        # Topologically Sorted Source Nodes: [attn_139], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf786, arg595_1, buf789, 6144, 48, grid=grid(6144), stream=stream0)
        del arg595_1
        buf790 = reinterpret_tensor(buf785, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf785  # reuse
        # Topologically Sorted Source Nodes: [x_857], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf779, arg594_1, buf790, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg594_1
        buf791 = reinterpret_tensor(buf784, (128, 48, 784), (37632, 784, 1), 0); del buf784  # reuse
        # Topologically Sorted Source Nodes: [x_857], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf789, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf790, (128, 48, 784), (37632, 784, 1), 0), out=buf791)
        buf792 = reinterpret_tensor(buf790, (8, 784, 768), (602112, 768, 1), 0); del buf790  # reuse
        # Topologically Sorted Source Nodes: [x_859], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf791, buf792, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf793 = reinterpret_tensor(buf791, (6272, 768), (768, 1), 0); del buf791  # reuse
        # Topologically Sorted Source Nodes: [x_859], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf792, (6272, 768), (768, 1), 0), reinterpret_tensor(arg596_1, (768, 768), (1, 768), 0), out=buf793)
        del arg596_1
        buf797 = buf792; del buf792  # reuse
        # Topologically Sorted Source Nodes: [x_859, mul_195, x_861, layer_norm_144], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf774, arg590_1, buf793, arg597_1, arg599_1, arg600_1, buf797, 6272, 768, grid=grid(6272), stream=stream0)
        del arg599_1
        del arg600_1
        # Topologically Sorted Source Nodes: [x_863], Original ATen: [aten.convolution]
        buf798 = extern_kernels.convolution(reinterpret_tensor(buf797, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg601_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf798, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg601_1
        del buf797
        buf799 = buf798; del buf798  # reuse
        # Topologically Sorted Source Nodes: [x_863, x_864, x_865], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf799, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg602_1
        del arg603_1
        del arg604_1
        del arg605_1
        del arg606_1
        # Topologically Sorted Source Nodes: [x_863, x_864, x_865, x_866], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf800 = extern_kernels.convolution(buf799, arg607_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf800, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg607_1
        buf801 = reinterpret_tensor(buf800, (8, 784, 768), (602112, 768, 1), 0); del buf800  # reuse
        buf805 = reinterpret_tensor(buf799, (8, 784, 768), (602112, 768, 1), 0); del buf799  # reuse
        # Topologically Sorted Source Nodes: [x_859, mul_195, x_861, mul_196, x_868, layer_norm_145], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf801, buf774, arg590_1, buf793, arg597_1, arg598_1, arg608_1, arg610_1, arg611_1, buf805, 6272, 768, grid=grid(6272), stream=stream0)
        del arg590_1
        del arg597_1
        del arg598_1
        del arg608_1
        del arg610_1
        del arg611_1
        buf806 = reinterpret_tensor(buf772, (6272, 3072), (3072, 1), 0); del buf772  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf805, (6272, 768), (768, 1), 0), reinterpret_tensor(arg612_1, (768, 3072), (1, 768), 0), out=buf806)
        del arg612_1
        buf807 = reinterpret_tensor(buf806, (8, 784, 3072), (2408448, 3072, 1), 0); del buf806  # reuse
        # Topologically Sorted Source Nodes: [x_870], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf807, arg613_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg613_1
        buf808 = reinterpret_tensor(buf805, (6272, 768), (768, 1), 0); del buf805  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf807, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg614_1, (3072, 768), (1, 3072), 0), out=buf808)
        del arg614_1
        buf812 = reinterpret_tensor(buf793, (8, 784, 768), (602112, 768, 1), 0); del buf793  # reuse
        # Topologically Sorted Source Nodes: [mul_197, x_874, layer_norm_146], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf801, arg609_1, buf808, arg615_1, arg617_1, arg618_1, buf812, 6272, 768, grid=grid(6272), stream=stream0)
        del arg617_1
        del arg618_1
        buf813 = buf779; del buf779  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf812, (6272, 768), (768, 1), 0), reinterpret_tensor(arg619_1, (768, 2304), (1, 768), 0), out=buf813)
        del arg619_1
        buf814 = buf782; del buf782  # reuse
        # Topologically Sorted Source Nodes: [q_97], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_10.run(buf813, arg620_1, buf814, 43008, 112, grid=grid(43008), stream=stream0)
        buf815 = buf783; del buf783  # reuse
        # Topologically Sorted Source Nodes: [q_97], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf814, buf815, 6144, 7, grid=grid(6144), stream=stream0)
        buf816 = buf814; del buf814  # reuse
        # Topologically Sorted Source Nodes: [k_97], Original ATen: [aten.linalg_vector_norm]
        triton_red_fused_linalg_vector_norm_12.run(buf813, arg620_1, buf816, 43008, 112, grid=grid(43008), stream=stream0)
        buf817 = buf781; del buf781  # reuse
        # Topologically Sorted Source Nodes: [k_97], Original ATen: [aten.linalg_vector_norm]
        triton_per_fused_linalg_vector_norm_11.run(buf816, buf817, 6144, 7, grid=grid(6144), stream=stream0)
        del buf816
        buf818 = reinterpret_tensor(buf812, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf812  # reuse
        # Topologically Sorted Source Nodes: [q_97, matmul_94], Original ATen: [aten.div, aten.clone]
        triton_poi_fused_clone_div_13.run(buf813, arg620_1, buf815, buf818, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del buf815
        buf819 = reinterpret_tensor(buf774, (8, 16, 784, 48), (602112, 37632, 48, 1), 0); del buf774  # reuse
        # Topologically Sorted Source Nodes: [matmul_94], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf813, arg620_1, buf817, buf819, 4816896, grid=grid(4816896), stream=stream0)
        buf820 = reinterpret_tensor(buf789, (128, 48, 48), (2304, 48, 1), 0); del buf789  # reuse
        # Topologically Sorted Source Nodes: [matmul_94], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf818, (128, 48, 784), (37632, 784, 1), 0), reinterpret_tensor(buf819, (128, 784, 48), (37632, 48, 1), 0), out=buf820)
        buf823 = reinterpret_tensor(buf786, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf786  # reuse
        # Topologically Sorted Source Nodes: [attn_142], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf820, arg621_1, buf823, 6144, 48, grid=grid(6144), stream=stream0)
        del arg621_1
        del buf820
        buf824 = reinterpret_tensor(buf819, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf819  # reuse
        # Topologically Sorted Source Nodes: [x_875], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf813, arg620_1, buf824, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del arg620_1
        del buf813
        buf825 = reinterpret_tensor(buf818, (128, 48, 784), (37632, 784, 1), 0); del buf818  # reuse
        # Topologically Sorted Source Nodes: [x_875], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf823, (128, 48, 48), (2304, 48, 1), 0), reinterpret_tensor(buf824, (128, 48, 784), (37632, 784, 1), 0), out=buf825)
        del buf823
        buf826 = reinterpret_tensor(buf824, (8, 784, 768), (602112, 768, 1), 0); del buf824  # reuse
        # Topologically Sorted Source Nodes: [x_877], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf825, buf826, 6272, 768, grid=grid(6272, 768), stream=stream0)
        buf827 = reinterpret_tensor(buf825, (6272, 768), (768, 1), 0); del buf825  # reuse
        # Topologically Sorted Source Nodes: [x_877], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf826, (6272, 768), (768, 1), 0), reinterpret_tensor(arg622_1, (768, 768), (1, 768), 0), out=buf827)
        del arg622_1
        buf828 = reinterpret_tensor(buf827, (8, 784, 768), (602112, 768, 1), 0); del buf827  # reuse
        buf832 = buf826; del buf826  # reuse
        # Topologically Sorted Source Nodes: [mul_197, x_874, x_877, mul_199, x_879, layer_norm_147], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_20.run(buf828, buf801, arg609_1, buf808, arg615_1, arg616_1, arg623_1, arg625_1, arg626_1, buf832, 6272, 768, grid=grid(6272), stream=stream0)
        del arg609_1
        del arg615_1
        del arg616_1
        del arg623_1
        del arg625_1
        del arg626_1
        del buf801
        del buf808
        # Topologically Sorted Source Nodes: [x_881], Original ATen: [aten.convolution]
        buf833 = extern_kernels.convolution(reinterpret_tensor(buf832, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), arg627_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf833, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg627_1
        del buf832
        buf834 = buf833; del buf833  # reuse
        # Topologically Sorted Source Nodes: [x_881, x_882, x_883], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_gelu_19.run(buf834, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg628_1
        del arg629_1
        del arg630_1
        del arg631_1
        del arg632_1
        # Topologically Sorted Source Nodes: [x_881, x_882, x_883, x_884], Original ATen: [aten.convolution, aten.gelu, aten._native_batch_norm_legit_no_training]
        buf835 = extern_kernels.convolution(buf834, arg633_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf835, (8, 768, 28, 28), (602112, 1, 21504, 768))
        del arg633_1
        buf839 = reinterpret_tensor(buf834, (8, 784, 768), (602112, 768, 1), 0); del buf834  # reuse
        # Topologically Sorted Source Nodes: [mul_200, x_886, layer_norm_148], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_18.run(buf828, arg624_1, buf835, arg634_1, arg636_1, arg637_1, buf839, 6272, 768, grid=grid(6272), stream=stream0)
        del arg636_1
        del arg637_1
        buf840 = reinterpret_tensor(buf807, (6272, 3072), (3072, 1), 0); del buf807  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf839, (6272, 768), (768, 1), 0), reinterpret_tensor(arg638_1, (768, 3072), (1, 768), 0), out=buf840)
        del arg638_1
        buf841 = reinterpret_tensor(buf840, (8, 784, 3072), (2408448, 3072, 1), 0); del buf840  # reuse
        # Topologically Sorted Source Nodes: [x_888], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf841, arg639_1, 19267584, grid=grid(19267584), stream=stream0)
        del arg639_1
        buf842 = reinterpret_tensor(buf839, (6272, 768), (768, 1), 0); del buf839  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf841, (6272, 3072), (3072, 1), 0), reinterpret_tensor(arg640_1, (3072, 768), (1, 3072), 0), out=buf842)
        del arg640_1
        del buf841
        buf843 = empty_strided_cuda((8, 785, 768), (602880, 768, 1), torch.float32)
        buf847 = empty_strided_cuda((8, 785, 768), (602880, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_893, x_norm1_2], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_22.run(arg642_1, buf828, arg624_1, buf835, arg634_1, arg635_1, buf842, arg641_1, arg643_1, arg644_1, buf843, buf847, 6280, 768, grid=grid(6280), stream=stream0)
        del arg624_1
        del arg634_1
        del arg635_1
        del arg641_1
        del arg642_1
        del arg643_1
        del arg644_1
        del buf828
        del buf835
        del buf842
        buf848 = reinterpret_tensor(buf817, (8, 768), (768, 1), 0); del buf817  # reuse
        # Topologically Sorted Source Nodes: [linear_205], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg646_1, reinterpret_tensor(buf847, (8, 768), (602880, 1), 0), reinterpret_tensor(arg645_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf848)
        del arg645_1
        del arg646_1
        buf849 = empty_strided_cuda((6280, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_206], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg648_1, reinterpret_tensor(buf847, (6280, 768), (768, 1), 0), reinterpret_tensor(arg647_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf849)
        del arg647_1
        del arg648_1
        buf850 = empty_strided_cuda((6280, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_207], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg650_1, reinterpret_tensor(buf847, (6280, 768), (768, 1), 0), reinterpret_tensor(arg649_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf850)
        del arg649_1
        del arg650_1
        # Topologically Sorted Source Nodes: [x_cls_8], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf851 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf848, (8, 16, 1, 48), (768, 48, 768, 1), 0), reinterpret_tensor(buf849, (8, 16, 785, 48), (602880, 48, 768, 1), 0), reinterpret_tensor(buf850, (8, 16, 785, 48), (602880, 48, 768, 1), 0), None, False)
        buf852 = buf851[0]
        del buf851
        buf856 = buf848; del buf848  # reuse
        # Topologically Sorted Source Nodes: [x_cls_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg652_1, reinterpret_tensor(buf852, (8, 768), (768, 1), 0), reinterpret_tensor(arg651_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf856)
        del arg651_1
        del arg652_1
        del buf852
        buf861 = reinterpret_tensor(buf850, (8, 785, 768), (602880, 768, 1), 0); del buf850  # reuse
        # Topologically Sorted Source Nodes: [x_attn_2, mul_202, x_894, x_895], Original ATen: [aten.cat, aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_mul_native_layer_norm_23.run(buf843, arg653_1, buf856, buf847, arg654_1, arg655_1, buf861, 6280, 768, grid=grid(6280), stream=stream0)
        del arg653_1
        del arg654_1
        del arg655_1
        buf862 = empty_strided_cuda((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_896], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf861, (8, 768), (602880, 1), 0), reinterpret_tensor(arg657_1, (768, 3072), (1, 768), 0), out=buf862)
        del arg657_1
        buf863 = reinterpret_tensor(buf862, (8, 1, 3072), (3072, 3072, 1), 0); del buf862  # reuse
        # Topologically Sorted Source Nodes: [x_896, x_897], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_24.run(buf863, arg658_1, 24576, grid=grid(24576), stream=stream0)
        del arg658_1
        buf864 = buf856; del buf856  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf863, (8, 3072), (3072, 1), 0), reinterpret_tensor(arg659_1, (3072, 768), (1, 3072), 0), out=buf864)
        del arg659_1
        buf865 = buf847; del buf847  # reuse
        buf869 = buf843; del buf843  # reuse
        # Topologically Sorted Source Nodes: [x_901, x_902, x_norm1_3], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_25.run(buf861, arg656_1, buf864, arg660_1, arg661_1, arg662_1, buf865, buf869, 6280, 768, grid=grid(6280), stream=stream0)
        del arg656_1
        del arg660_1
        del arg661_1
        del arg662_1
        buf870 = buf864; del buf864  # reuse
        # Topologically Sorted Source Nodes: [linear_211], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg664_1, reinterpret_tensor(buf869, (8, 768), (602880, 1), 0), reinterpret_tensor(arg663_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf870)
        del arg663_1
        del arg664_1
        buf871 = reinterpret_tensor(buf861, (6280, 768), (768, 1), 0); del buf861  # reuse
        # Topologically Sorted Source Nodes: [linear_212], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg666_1, reinterpret_tensor(buf869, (6280, 768), (768, 1), 0), reinterpret_tensor(arg665_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf871)
        del arg665_1
        del arg666_1
        buf872 = buf849; del buf849  # reuse
        # Topologically Sorted Source Nodes: [linear_213], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg668_1, reinterpret_tensor(buf869, (6280, 768), (768, 1), 0), reinterpret_tensor(arg667_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf872)
        del arg667_1
        del arg668_1
        # Topologically Sorted Source Nodes: [x_cls_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf873 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf870, (8, 16, 1, 48), (768, 48, 768, 1), 0), reinterpret_tensor(buf871, (8, 16, 785, 48), (602880, 48, 768, 1), 0), reinterpret_tensor(buf872, (8, 16, 785, 48), (602880, 48, 768, 1), 0), None, False)
        del buf871
        buf874 = buf873[0]
        del buf873
        buf878 = buf870; del buf870  # reuse
        # Topologically Sorted Source Nodes: [x_cls_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg670_1, reinterpret_tensor(buf874, (8, 768), (768, 1), 0), reinterpret_tensor(arg669_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf878)
        del arg669_1
        del arg670_1
        del buf874
        buf883 = reinterpret_tensor(buf872, (8, 785, 768), (602880, 768, 1), 0); del buf872  # reuse
        # Topologically Sorted Source Nodes: [x_attn_3, mul_204, x_903, x_904], Original ATen: [aten.cat, aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_mul_native_layer_norm_23.run(buf865, arg671_1, buf878, buf869, arg672_1, arg673_1, buf883, 6280, 768, grid=grid(6280), stream=stream0)
        del arg671_1
        del arg672_1
        del arg673_1
        del buf865
        buf884 = reinterpret_tensor(buf863, (8, 3072), (3072, 1), 0); del buf863  # reuse
        # Topologically Sorted Source Nodes: [x_905], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf883, (8, 768), (602880, 1), 0), reinterpret_tensor(arg675_1, (768, 3072), (1, 768), 0), out=buf884)
        del arg675_1
        buf885 = reinterpret_tensor(buf884, (8, 1, 3072), (3072, 3072, 1), 0); del buf884  # reuse
        # Topologically Sorted Source Nodes: [x_905, x_906], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_24.run(buf885, arg676_1, 24576, grid=grid(24576), stream=stream0)
        del arg676_1
        buf886 = buf878; del buf878  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf885, (8, 3072), (3072, 1), 0), reinterpret_tensor(arg677_1, (3072, 768), (1, 3072), 0), out=buf886)
        del arg677_1
        del buf885
        buf887 = buf869; del buf869  # reuse
        buf888 = empty_strided_cuda((8, 785, 1), (785, 1, 6304), torch.float32)
        buf889 = empty_strided_cuda((8, 785, 1), (785, 1, 6304), torch.float32)
        # Topologically Sorted Source Nodes: [x_910, x_911, x_912], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_26.run(buf883, arg674_1, buf886, arg678_1, buf887, buf888, buf889, 6280, 768, grid=grid(6280), stream=stream0)
        del arg674_1
        del arg678_1
        del buf883
        buf891 = buf886; del buf886  # reuse
        # Topologically Sorted Source Nodes: [x_914], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf887, buf888, buf889, arg679_1, arg680_1, buf891, 6144, grid=grid(6144), stream=stream0)
        del arg679_1
        del arg680_1
        del buf887
        del buf888
        del buf889
        buf892 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_914, x_915], Original ATen: [aten.clone, aten.addmm]
        extern_kernels.addmm(arg682_1, buf891, reinterpret_tensor(arg681_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf892)
        del arg681_1
        del arg682_1
        del buf891
    return (buf892, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((192, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('xcit_large_24_p8_224', benchmark_compiled_module)
