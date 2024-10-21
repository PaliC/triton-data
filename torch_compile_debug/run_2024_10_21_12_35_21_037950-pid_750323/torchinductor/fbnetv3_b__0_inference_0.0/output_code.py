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


# kernel path: /tmp/torchinductor_sahanp/eh/ceht27vbsmlt6ox5m7xhqumvk5yvhnv6w4p3xfjvfagojzfql7rr.py
# Topologically Sorted Source Nodes: [x_278], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_278 => convolution_124
# Graph fragment:
#   %convolution_124 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (65536*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (196608*y1)), tmp0, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lt/cltiu3szd73oc7uccpazylbbmynvcroh24nhc4luunkmkbitxhos.py
# Topologically Sorted Source Nodes: [x_278], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_278 => convolution_124
# Graph fragment:
#   %convolution_124 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/wp/cwpwudminc7zt47gdcknywdd653dqn3ogrh25mhuqusjuvxbf2st.py
# Topologically Sorted Source Nodes: [x_279, x_280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_279 => add_293, mul_357, mul_358, sub_87
#   x_280 => add_294, clamp_max_95, clamp_min_95, div_95, mul_359
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_124, %unsqueeze_697), kwargs = {})
#   %mul_357 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_358 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_357, %unsqueeze_701), kwargs = {})
#   %add_293 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_358, %unsqueeze_703), kwargs = {})
#   %add_294 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_293, 3), kwargs = {})
#   %clamp_min_95 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_294, 0), kwargs = {})
#   %clamp_max_95 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_95, 6), kwargs = {})
#   %mul_359 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_293, %clamp_max_95), kwargs = {})
#   %div_95 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_359, 6), kwargs = {})
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
    xnumel = 2097152
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


# kernel path: /tmp/torchinductor_sahanp/zz/czzoqjhkg4est3hlbixxmzh5lsdf3hrxsfn6kkudxhgz54olswtw.py
# Topologically Sorted Source Nodes: [x_285, x_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_285 => add_299, mul_365, mul_366, sub_89
#   x_286 => add_300
# Graph fragment:
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_126, %unsqueeze_713), kwargs = {})
#   %mul_365 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %unsqueeze_715), kwargs = {})
#   %mul_366 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_365, %unsqueeze_717), kwargs = {})
#   %add_299 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_366, %unsqueeze_719), kwargs = {})
#   %add_300 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_299, %div_95), kwargs = {})
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
    xnumel = 2097152
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


# kernel path: /tmp/torchinductor_sahanp/hy/chys6i2frgp4sllbxwz2po5y7k4zpi6ceb3jopabpo3qpdn2m2eu.py
# Topologically Sorted Source Nodes: [x_291, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_291 => add_305, mul_372, mul_373, sub_91
#   x_292 => add_306
# Graph fragment:
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_128, %unsqueeze_729), kwargs = {})
#   %mul_372 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %unsqueeze_731), kwargs = {})
#   %mul_373 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_372, %unsqueeze_733), kwargs = {})
#   %add_305 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_373, %unsqueeze_735), kwargs = {})
#   %add_306 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_305, %add_300), kwargs = {})
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
    xnumel = 2097152
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2v/c2vk5z5ooagr3sl4dcavic4ru52gftn3amjmntp6xxaaxwbzlvkf.py
# Topologically Sorted Source Nodes: [x_294, x_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_294 => add_308, mul_375, mul_376, sub_92
#   x_295 => add_309, clamp_max_98, clamp_min_98, div_98, mul_377
# Graph fragment:
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_129, %unsqueeze_737), kwargs = {})
#   %mul_375 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_376 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_375, %unsqueeze_741), kwargs = {})
#   %add_308 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_376, %unsqueeze_743), kwargs = {})
#   %add_309 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_308, 3), kwargs = {})
#   %clamp_min_98 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_309, 0), kwargs = {})
#   %clamp_max_98 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_98, 6), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_308, %clamp_max_98), kwargs = {})
#   %div_98 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_377, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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


# kernel path: /tmp/torchinductor_sahanp/g7/cg7v336eti7klxb2j35xqheowiymbb6hox2ichbibe4ttx45mixd.py
# Topologically Sorted Source Nodes: [x_297, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_297 => add_311, mul_379, mul_380, sub_93
#   x_298 => add_312, clamp_max_99, clamp_min_99, div_99, mul_381
# Graph fragment:
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_130, %unsqueeze_745), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_747), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_379, %unsqueeze_749), kwargs = {})
#   %add_311 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_380, %unsqueeze_751), kwargs = {})
#   %add_312 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_311, 3), kwargs = {})
#   %clamp_min_99 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_312, 0), kwargs = {})
#   %clamp_max_99 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_99, 6), kwargs = {})
#   %mul_381 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_311, %clamp_max_99), kwargs = {})
#   %div_99 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_381, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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


# kernel path: /tmp/torchinductor_sahanp/vv/cvvltneszfidcid2hd4jsfwmijetxdywg5tlz67jr4beqf7zqlqo.py
# Topologically Sorted Source Nodes: [x_300], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_300 => add_314, mul_383, mul_384, sub_94
# Graph fragment:
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_131, %unsqueeze_753), kwargs = {})
#   %mul_383 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %mul_384 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_383, %unsqueeze_757), kwargs = {})
#   %add_314 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_384, %unsqueeze_759), kwargs = {})
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
    xnumel = 786432
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


# kernel path: /tmp/torchinductor_sahanp/ar/car5zft7qpm3tt7refv2i3yzrdyyngquo2o54pgxnyuypk6falj6.py
# Topologically Sorted Source Nodes: [x_302, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_302 => add_316, mul_386, mul_387, sub_95
#   x_303 => add_317, clamp_max_100, clamp_min_100, div_100, mul_388
# Graph fragment:
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_132, %unsqueeze_761), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %unsqueeze_763), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_386, %unsqueeze_765), kwargs = {})
#   %add_316 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_387, %unsqueeze_767), kwargs = {})
#   %add_317 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_316, 3), kwargs = {})
#   %clamp_min_100 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_317, 0), kwargs = {})
#   %clamp_max_100 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_100, 6), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_316, %clamp_max_100), kwargs = {})
#   %div_100 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_388, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 48
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


# kernel path: /tmp/torchinductor_sahanp/24/c24qk7dvppmhdzer4rwl7dbcg67bihikpoaewlyogix27zoy3l46.py
# Topologically Sorted Source Nodes: [x_308, x_309], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_308 => add_322, mul_394, mul_395, sub_97
#   x_309 => add_323
# Graph fragment:
#   %sub_97 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_134, %unsqueeze_777), kwargs = {})
#   %mul_394 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_97, %unsqueeze_779), kwargs = {})
#   %mul_395 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_394, %unsqueeze_781), kwargs = {})
#   %add_322 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_395, %unsqueeze_783), kwargs = {})
#   %add_323 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_322, %add_314), kwargs = {})
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
    xnumel = 786432
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


# kernel path: /tmp/torchinductor_sahanp/qg/cqgdoaymznd4ll3ywnufz753ulcjc5ardpnspbtom7bwwpixedzb.py
# Topologically Sorted Source Nodes: [x_329, x_330], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_329 => add_343, mul_419, mul_420, sub_104
#   x_330 => add_344, clamp_max_106, clamp_min_106, div_106, mul_421
# Graph fragment:
#   %sub_104 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_141, %unsqueeze_833), kwargs = {})
#   %mul_419 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_104, %unsqueeze_835), kwargs = {})
#   %mul_420 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_419, %unsqueeze_837), kwargs = {})
#   %add_343 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_420, %unsqueeze_839), kwargs = {})
#   %add_344 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_343, 3), kwargs = {})
#   %clamp_min_106 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_344, 0), kwargs = {})
#   %clamp_max_106 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_106, 6), kwargs = {})
#   %mul_421 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_343, %clamp_max_106), kwargs = {})
#   %div_106 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_421, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3932160
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


# kernel path: /tmp/torchinductor_sahanp/er/cermrcpo5o7mre6md3pd345njne5k6nqdsxlbnfty2243ngnfw37.py
# Topologically Sorted Source Nodes: [x_332], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_332 => add_346, mul_423, mul_424, sub_105
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_142, %unsqueeze_841), kwargs = {})
#   %mul_423 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_424 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_423, %unsqueeze_845), kwargs = {})
#   %add_346 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_424, %unsqueeze_847), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
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


# kernel path: /tmp/torchinductor_sahanp/qw/cqwk6evi6n5xvtyzkmhfkbmvmqa63i3oyjpd7zcdrlu6thsxr26q.py
# Topologically Sorted Source Nodes: [x_333, x_se_72], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_333 => add_347, clamp_max_107, clamp_min_107, div_107, mul_425
#   x_se_72 => mean_19
# Graph fragment:
#   %add_347 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_346, 3), kwargs = {})
#   %clamp_min_107 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_347, 0), kwargs = {})
#   %clamp_max_107 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_107, 6), kwargs = {})
#   %mul_425 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_346, %clamp_max_107), kwargs = {})
#   %div_107 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_425, 6), kwargs = {})
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_107, [2, 3], True), kwargs = {})
triton_red_fused_hardswish_mean_12 = async_compile.triton('triton_red_fused_hardswish_mean_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_mean_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_hardswish_mean_12(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 120
    x1 = (xindex // 120)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/mq/cmqta557pzgvfteuw5hsldn37zznb2an25l7qjif2awyfwwwmnwl.py
# Topologically Sorted Source Nodes: [x_333, x_se_72], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_333 => add_347, clamp_max_107, clamp_min_107, div_107, mul_425
#   x_se_72 => mean_19
# Graph fragment:
#   %add_347 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_346, 3), kwargs = {})
#   %clamp_min_107 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_347, 0), kwargs = {})
#   %clamp_max_107 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_107, 6), kwargs = {})
#   %mul_425 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_346, %clamp_max_107), kwargs = {})
#   %div_107 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_425, 6), kwargs = {})
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_107, [2, 3], True), kwargs = {})
triton_per_fused_hardswish_mean_13 = async_compile.triton('triton_per_fused_hardswish_mean_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_13(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 120
    x1 = (xindex // 120)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (960*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 1024.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nw/cnwj5dkmvdq2dirlc2odfizhlmbnutm2vc4gpkbs7lkztli5onzu.py
# Topologically Sorted Source Nodes: [x_333, x_se_72, x_se_73, x_se_74], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_333 => add_347, clamp_max_107, clamp_min_107, div_107, mul_425
#   x_se_72 => mean_19
#   x_se_73 => convolution_143
#   x_se_74 => add_348, clamp_max_108, clamp_min_108, div_108, mul_426
# Graph fragment:
#   %add_347 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_346, 3), kwargs = {})
#   %clamp_min_107 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_347, 0), kwargs = {})
#   %clamp_max_107 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_107, 6), kwargs = {})
#   %mul_425 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_346, %clamp_max_107), kwargs = {})
#   %div_107 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_425, 6), kwargs = {})
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_107, [2, 3], True), kwargs = {})
#   %convolution_143 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_19, %arg96_1, %arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_348 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_143, 3), kwargs = {})
#   %clamp_min_108 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_348, 0), kwargs = {})
#   %clamp_max_108 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_108, 6), kwargs = {})
#   %mul_426 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, %clamp_max_108), kwargs = {})
#   %div_108 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_426, 6), kwargs = {})
triton_poi_fused_convolution_hardswish_mean_14 = async_compile.triton('triton_poi_fused_convolution_hardswish_mean_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardswish_mean_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
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


# kernel path: /tmp/torchinductor_sahanp/2o/c2obxjuvh3rfocszv7z5wg3xglukmsurpr5jvfyglnl6notcwkr4.py
# Topologically Sorted Source Nodes: [x_333, x_se_72, x_se_73, x_se_74, x_se_75, hardsigmoid_18, x_334], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_18 => add_349, clamp_max_109, clamp_min_109, div_109
#   x_333 => add_347, clamp_max_107, clamp_min_107, div_107, mul_425
#   x_334 => mul_427
#   x_se_72 => mean_19
#   x_se_73 => convolution_143
#   x_se_74 => add_348, clamp_max_108, clamp_min_108, div_108, mul_426
#   x_se_75 => convolution_144
# Graph fragment:
#   %add_347 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_346, 3), kwargs = {})
#   %clamp_min_107 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_347, 0), kwargs = {})
#   %clamp_max_107 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_107, 6), kwargs = {})
#   %mul_425 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_346, %clamp_max_107), kwargs = {})
#   %div_107 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_425, 6), kwargs = {})
#   %mean_19 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_107, [2, 3], True), kwargs = {})
#   %convolution_143 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_19, %arg96_1, %arg97_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_348 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_143, 3), kwargs = {})
#   %clamp_min_108 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_348, 0), kwargs = {})
#   %clamp_max_108 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_108, 6), kwargs = {})
#   %mul_426 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_143, %clamp_max_108), kwargs = {})
#   %div_108 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_426, 6), kwargs = {})
#   %convolution_144 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_108, %arg98_1, %arg99_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_349 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_144, 3), kwargs = {})
#   %clamp_min_109 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_349, 0), kwargs = {})
#   %clamp_max_109 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_109, 6), kwargs = {})
#   %div_109 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_109, 6), kwargs = {})
#   %mul_427 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_107, %div_109), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_15 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_15(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 120
    x2 = (xindex // 122880)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr0 + (x0 + (120*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fu/cfuht6lvgc2w7gdbk4rq7u37nr6jowb4nvcssgq4zeyhkbzvzyeg.py
# Topologically Sorted Source Nodes: [x_336], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_336 => add_351, mul_429, mul_430, sub_106
# Graph fragment:
#   %sub_106 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_145, %unsqueeze_849), kwargs = {})
#   %mul_429 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_106, %unsqueeze_851), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_429, %unsqueeze_853), kwargs = {})
#   %add_351 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_430, %unsqueeze_855), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 40
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


# kernel path: /tmp/torchinductor_sahanp/3j/c3jsnwq2etegkrw2n5hbfo6bhd3ad22uafnu44hlk6qcjsjpe2dw.py
# Topologically Sorted Source Nodes: [x_338, x_339], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_338 => add_353, mul_432, mul_433, sub_107
#   x_339 => add_354, clamp_max_110, clamp_min_110, div_110, mul_434
# Graph fragment:
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_146, %unsqueeze_857), kwargs = {})
#   %mul_432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %unsqueeze_859), kwargs = {})
#   %mul_433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_432, %unsqueeze_861), kwargs = {})
#   %add_353 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_433, %unsqueeze_863), kwargs = {})
#   %add_354 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_353, 3), kwargs = {})
#   %clamp_min_110 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_354, 0), kwargs = {})
#   %clamp_max_110 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_110, 6), kwargs = {})
#   %mul_434 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_353, %clamp_max_110), kwargs = {})
#   %div_110 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_434, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 983040
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


# kernel path: /tmp/torchinductor_sahanp/uu/cuult2jfux77c47d5qhsww3al2sumu4i4hmgal65kyzubynhjxg3.py
# Topologically Sorted Source Nodes: [x_342, x_se_76, x_se_77, x_se_78], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_342 => add_357, clamp_max_111, clamp_min_111, div_111, mul_438
#   x_se_76 => mean_20
#   x_se_77 => convolution_148
#   x_se_78 => add_358, clamp_max_112, clamp_min_112, div_112, mul_439
# Graph fragment:
#   %add_357 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_356, 3), kwargs = {})
#   %clamp_min_111 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_357, 0), kwargs = {})
#   %clamp_max_111 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_111, 6), kwargs = {})
#   %mul_438 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_356, %clamp_max_111), kwargs = {})
#   %div_111 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_438, 6), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_111, [2, 3], True), kwargs = {})
#   %convolution_148 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_20, %arg115_1, %arg116_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_358 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_148, 3), kwargs = {})
#   %clamp_min_112 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_358, 0), kwargs = {})
#   %clamp_max_112 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_112, 6), kwargs = {})
#   %mul_439 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_148, %clamp_max_112), kwargs = {})
#   %div_112 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_439, 6), kwargs = {})
triton_poi_fused_convolution_hardswish_mean_18 = async_compile.triton('triton_poi_fused_convolution_hardswish_mean_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardswish_mean_18(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
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


# kernel path: /tmp/torchinductor_sahanp/xs/cxsneg2qhheb6w6juzigtkmtdvjvldutb3onj5e6rnzwgxei2v44.py
# Topologically Sorted Source Nodes: [x_345, x_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_345 => add_361, mul_442, mul_443, sub_109
#   x_346 => add_362
# Graph fragment:
#   %sub_109 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_150, %unsqueeze_873), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_109, %unsqueeze_875), kwargs = {})
#   %mul_443 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_442, %unsqueeze_877), kwargs = {})
#   %add_361 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_443, %unsqueeze_879), kwargs = {})
#   %add_362 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_361, %add_351), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 40
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


# kernel path: /tmp/torchinductor_sahanp/zc/czcgrciitvj6strjbxsasjv4sdgofprnsxonmgqzlrl7vknjljtr.py
# Topologically Sorted Source Nodes: [x_365, x_366], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_365 => add_383, mul_468, mul_469, sub_115
#   x_366 => add_384
# Graph fragment:
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_160, %unsqueeze_921), kwargs = {})
#   %mul_468 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %unsqueeze_923), kwargs = {})
#   %mul_469 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_468, %unsqueeze_925), kwargs = {})
#   %add_383 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_469, %unsqueeze_927), kwargs = {})
#   %add_384 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_383, %add_373), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 40
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/so/csorcxniw7swz646kvnb54zry3jhqeewg7bbsfrtsnxgrsag26sn.py
# Topologically Sorted Source Nodes: [x_378, x_379], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_378 => add_397, mul_484, mul_485, sub_119
#   x_379 => add_398, clamp_max_126, clamp_min_126, div_126, mul_486
# Graph fragment:
#   %sub_119 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_166, %unsqueeze_953), kwargs = {})
#   %mul_484 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_119, %unsqueeze_955), kwargs = {})
#   %mul_485 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_484, %unsqueeze_957), kwargs = {})
#   %add_397 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_485, %unsqueeze_959), kwargs = {})
#   %add_398 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_397, 3), kwargs = {})
#   %clamp_min_126 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_398, 0), kwargs = {})
#   %clamp_max_126 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_126, 6), kwargs = {})
#   %mul_486 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_397, %clamp_max_126), kwargs = {})
#   %div_126 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_486, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1638400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 200
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


# kernel path: /tmp/torchinductor_sahanp/o7/co7d5jfp7er7hj4duxxam4euwzmsmq5de7huhevnafxmhss7eva2.py
# Topologically Sorted Source Nodes: [x_381, x_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_381 => add_400, mul_488, mul_489, sub_120
#   x_382 => add_401, clamp_max_127, clamp_min_127, div_127, mul_490
# Graph fragment:
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_167, %unsqueeze_961), kwargs = {})
#   %mul_488 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_120, %unsqueeze_963), kwargs = {})
#   %mul_489 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_488, %unsqueeze_965), kwargs = {})
#   %add_400 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_489, %unsqueeze_967), kwargs = {})
#   %add_401 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_400, 3), kwargs = {})
#   %clamp_min_127 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_401, 0), kwargs = {})
#   %clamp_max_127 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_127, 6), kwargs = {})
#   %mul_490 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_400, %clamp_max_127), kwargs = {})
#   %div_127 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_490, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 409600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 200
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


# kernel path: /tmp/torchinductor_sahanp/jw/cjwxvp2jqcqepx2bgw6po7jzp6oybfoqaeev4fdptvevciuhm6nt.py
# Topologically Sorted Source Nodes: [x_384], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_384 => add_403, mul_492, mul_493, sub_121
# Graph fragment:
#   %sub_121 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_168, %unsqueeze_969), kwargs = {})
#   %mul_492 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_121, %unsqueeze_971), kwargs = {})
#   %mul_493 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_492, %unsqueeze_973), kwargs = {})
#   %add_403 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_493, %unsqueeze_975), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/b6/cb65ggf7yvtibfold252j57kyheahzlvsolihwss7ngoul4anrm4.py
# Topologically Sorted Source Nodes: [x_386, x_387], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_386 => add_405, mul_495, mul_496, sub_122
#   x_387 => add_406, clamp_max_128, clamp_min_128, div_128, mul_497
# Graph fragment:
#   %sub_122 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_169, %unsqueeze_977), kwargs = {})
#   %mul_495 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_122, %unsqueeze_979), kwargs = {})
#   %mul_496 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_495, %unsqueeze_981), kwargs = {})
#   %add_405 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_496, %unsqueeze_983), kwargs = {})
#   %add_406 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_405, 3), kwargs = {})
#   %clamp_min_128 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_406, 0), kwargs = {})
#   %clamp_max_128 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_128, 6), kwargs = {})
#   %mul_497 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_405, %clamp_max_128), kwargs = {})
#   %div_128 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_497, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 216
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


# kernel path: /tmp/torchinductor_sahanp/fz/cfztlppmi2doaquriowtmz4yappz45m4xiimoxdswvgnx5tj4y3w.py
# Topologically Sorted Source Nodes: [x_392, x_393], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_392 => add_411, mul_503, mul_504, sub_124
#   x_393 => add_412
# Graph fragment:
#   %sub_124 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_171, %unsqueeze_993), kwargs = {})
#   %mul_503 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_124, %unsqueeze_995), kwargs = {})
#   %mul_504 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_503, %unsqueeze_997), kwargs = {})
#   %add_411 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_504, %unsqueeze_999), kwargs = {})
#   %add_412 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_411, %add_403), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 72
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


# kernel path: /tmp/torchinductor_sahanp/l4/cl4f353hlik6taaakxvsbgxcu5gizovvzugo7bleu7a54uqj23s4.py
# Topologically Sorted Source Nodes: [x_422, x_423], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_422 => add_441, mul_539, mul_540, sub_134
#   x_423 => add_442, clamp_max_136, clamp_min_136, div_136, mul_541
# Graph fragment:
#   %sub_134 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_181, %unsqueeze_1073), kwargs = {})
#   %mul_539 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_134, %unsqueeze_1075), kwargs = {})
#   %mul_540 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_539, %unsqueeze_1077), kwargs = {})
#   %add_441 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_540, %unsqueeze_1079), kwargs = {})
#   %add_442 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_441, 3), kwargs = {})
#   %clamp_min_136 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_442, 0), kwargs = {})
#   %clamp_max_136 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_136, 6), kwargs = {})
#   %mul_541 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_441, %clamp_max_136), kwargs = {})
#   %div_136 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_541, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 737280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 360
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


# kernel path: /tmp/torchinductor_sahanp/nv/cnvdvshpeixuhcxh7v4dymgs4eg2gklkvdgy6ie5xxtnvzglsvoj.py
# Topologically Sorted Source Nodes: [x_425], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_425 => add_444, mul_543, mul_544, sub_135
# Graph fragment:
#   %sub_135 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_182, %unsqueeze_1081), kwargs = {})
#   %mul_543 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_135, %unsqueeze_1083), kwargs = {})
#   %mul_544 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_543, %unsqueeze_1085), kwargs = {})
#   %add_444 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_544, %unsqueeze_1087), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 737280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 360
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


# kernel path: /tmp/torchinductor_sahanp/pd/cpdhrvnhnjoi65bunodflqc7keyqxjzjevi4ut75tvkmlw4uvkfe.py
# Topologically Sorted Source Nodes: [x_426, x_se_92], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_426 => add_445, clamp_max_137, clamp_min_137, div_137, mul_545
#   x_se_92 => mean_24
# Graph fragment:
#   %add_445 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_444, 3), kwargs = {})
#   %clamp_min_137 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_445, 0), kwargs = {})
#   %clamp_max_137 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_137, 6), kwargs = {})
#   %mul_545 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_444, %clamp_max_137), kwargs = {})
#   %div_137 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_545, 6), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_137, [2, 3], True), kwargs = {})
triton_red_fused_hardswish_mean_28 = async_compile.triton('triton_red_fused_hardswish_mean_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_mean_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_hardswish_mean_28(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5760
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 360
    x1 = (xindex // 360)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (360*r2) + (46080*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/mv/cmvnv3kwsdaidod2r4xlo5qnjhgcaafwf4vlh77nqp23yzccd7cb.py
# Topologically Sorted Source Nodes: [x_426, x_se_92], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_426 => add_445, clamp_max_137, clamp_min_137, div_137, mul_545
#   x_se_92 => mean_24
# Graph fragment:
#   %add_445 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_444, 3), kwargs = {})
#   %clamp_min_137 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_445, 0), kwargs = {})
#   %clamp_max_137 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_137, 6), kwargs = {})
#   %mul_545 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_444, %clamp_max_137), kwargs = {})
#   %div_137 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_545, 6), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_137, [2, 3], True), kwargs = {})
triton_per_fused_hardswish_mean_29 = async_compile.triton('triton_per_fused_hardswish_mean_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_29(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2880
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 360
    x1 = (xindex // 360)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (360*r2) + (720*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 256.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/y4/cy4llqlczfvyqgg7nn4g6vmxh7dbqvsjk5heuktv5talssr52osr.py
# Topologically Sorted Source Nodes: [x_426, x_se_92, x_se_93, x_se_94], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_426 => add_445, clamp_max_137, clamp_min_137, div_137, mul_545
#   x_se_92 => mean_24
#   x_se_93 => convolution_183
#   x_se_94 => add_446, clamp_max_138, clamp_min_138, div_138, mul_546
# Graph fragment:
#   %add_445 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_444, 3), kwargs = {})
#   %clamp_min_137 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_445, 0), kwargs = {})
#   %clamp_max_137 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_137, 6), kwargs = {})
#   %mul_545 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_444, %clamp_max_137), kwargs = {})
#   %div_137 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_545, 6), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_137, [2, 3], True), kwargs = {})
#   %convolution_183 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_24, %arg266_1, %arg267_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_446 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_183, 3), kwargs = {})
#   %clamp_min_138 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_446, 0), kwargs = {})
#   %clamp_max_138 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_138, 6), kwargs = {})
#   %mul_546 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_183, %clamp_max_138), kwargs = {})
#   %div_138 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_546, 6), kwargs = {})
triton_poi_fused_convolution_hardswish_mean_30 = async_compile.triton('triton_poi_fused_convolution_hardswish_mean_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardswish_mean_30(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
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


# kernel path: /tmp/torchinductor_sahanp/jt/cjtskixeyfhaup33o4wargegitqf6s2pydiscumdx6jcehe66km4.py
# Topologically Sorted Source Nodes: [x_426, x_se_92, x_se_93, x_se_94, x_se_95, hardsigmoid_23, x_427], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_23 => add_447, clamp_max_139, clamp_min_139, div_139
#   x_426 => add_445, clamp_max_137, clamp_min_137, div_137, mul_545
#   x_427 => mul_547
#   x_se_92 => mean_24
#   x_se_93 => convolution_183
#   x_se_94 => add_446, clamp_max_138, clamp_min_138, div_138, mul_546
#   x_se_95 => convolution_184
# Graph fragment:
#   %add_445 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_444, 3), kwargs = {})
#   %clamp_min_137 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_445, 0), kwargs = {})
#   %clamp_max_137 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_137, 6), kwargs = {})
#   %mul_545 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_444, %clamp_max_137), kwargs = {})
#   %div_137 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_545, 6), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_137, [2, 3], True), kwargs = {})
#   %convolution_183 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_24, %arg266_1, %arg267_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_446 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_183, 3), kwargs = {})
#   %clamp_min_138 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_446, 0), kwargs = {})
#   %clamp_max_138 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_138, 6), kwargs = {})
#   %mul_546 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_183, %clamp_max_138), kwargs = {})
#   %div_138 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_546, 6), kwargs = {})
#   %convolution_184 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_138, %arg268_1, %arg269_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_447 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_184, 3), kwargs = {})
#   %clamp_min_139 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_447, 0), kwargs = {})
#   %clamp_max_139 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_139, 6), kwargs = {})
#   %div_139 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_139, 6), kwargs = {})
#   %mul_547 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_137, %div_139), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 737280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 360
    x2 = (xindex // 92160)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr0 + (x0 + (360*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bh/cbh6lcm7qhebgamd7fhjj5fwtcoe2yk74oobdd46666crsj36aj4.py
# Topologically Sorted Source Nodes: [x_429], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_429 => add_449, mul_549, mul_550, sub_136
# Graph fragment:
#   %sub_136 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_185, %unsqueeze_1089), kwargs = {})
#   %mul_549 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_136, %unsqueeze_1091), kwargs = {})
#   %mul_550 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_549, %unsqueeze_1093), kwargs = {})
#   %add_449 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_550, %unsqueeze_1095), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 245760
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


# kernel path: /tmp/torchinductor_sahanp/pr/cprtnwtawpyq5d2mq76og3w2kwpn6s7lvfyhnzhpv7bpvyaob33o.py
# Topologically Sorted Source Nodes: [x_435, x_se_96, x_se_97, x_se_98], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_435 => add_455, clamp_max_141, clamp_min_141, div_141, mul_558
#   x_se_96 => mean_25
#   x_se_97 => convolution_188
#   x_se_98 => add_456, clamp_max_142, clamp_min_142, div_142, mul_559
# Graph fragment:
#   %add_455 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_454, 3), kwargs = {})
#   %clamp_min_141 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_455, 0), kwargs = {})
#   %clamp_max_141 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_141, 6), kwargs = {})
#   %mul_558 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_454, %clamp_max_141), kwargs = {})
#   %div_141 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_558, 6), kwargs = {})
#   %mean_25 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_141, [2, 3], True), kwargs = {})
#   %convolution_188 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_25, %arg285_1, %arg286_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_456 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_188, 3), kwargs = {})
#   %clamp_min_142 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_456, 0), kwargs = {})
#   %clamp_max_142 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_142, 6), kwargs = {})
#   %mul_559 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_188, %clamp_max_142), kwargs = {})
#   %div_142 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_559, 6), kwargs = {})
triton_poi_fused_convolution_hardswish_mean_33 = async_compile.triton('triton_poi_fused_convolution_hardswish_mean_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardswish_mean_33(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
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


# kernel path: /tmp/torchinductor_sahanp/54/c54m2km33sjjbpbdbpagle5pnjcrlrtagdx7qlg6ytbheqhntgzx.py
# Topologically Sorted Source Nodes: [x_438, x_439], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_438 => add_459, mul_562, mul_563, sub_139
#   x_439 => add_460
# Graph fragment:
#   %sub_139 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_190, %unsqueeze_1113), kwargs = {})
#   %mul_562 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_139, %unsqueeze_1115), kwargs = {})
#   %mul_563 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_562, %unsqueeze_1117), kwargs = {})
#   %add_459 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_563, %unsqueeze_1119), kwargs = {})
#   %add_460 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_459, %add_449), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 245760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 120
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


# kernel path: /tmp/torchinductor_sahanp/aq/caqxytvdcndtca67fjill5senhkayjbrnf4oorlyi5dro746vewa.py
# Topologically Sorted Source Nodes: [x_481, x_482], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_481 => add_506, mul_617, mul_618, sub_152
#   x_482 => add_507, clamp_max_160, clamp_min_160, div_160, mul_619
# Graph fragment:
#   %sub_152 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_211, %unsqueeze_1217), kwargs = {})
#   %mul_617 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_152, %unsqueeze_1219), kwargs = {})
#   %mul_618 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_617, %unsqueeze_1221), kwargs = {})
#   %add_506 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_618, %unsqueeze_1223), kwargs = {})
#   %add_507 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_506, 3), kwargs = {})
#   %clamp_min_160 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_507, 0), kwargs = {})
#   %clamp_max_160 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_160, 6), kwargs = {})
#   %mul_619 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_506, %clamp_max_160), kwargs = {})
#   %div_160 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_619, 6), kwargs = {})
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
    xnumel = 1474560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 720
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


# kernel path: /tmp/torchinductor_sahanp/nl/cnlsvje6cca55dumvo3tehnelhhuiowamqddmvjqensnonkyxl4b.py
# Topologically Sorted Source Nodes: [x_484], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_484 => add_509, mul_621, mul_622, sub_153
# Graph fragment:
#   %sub_153 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_212, %unsqueeze_1225), kwargs = {})
#   %mul_621 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_153, %unsqueeze_1227), kwargs = {})
#   %mul_622 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_621, %unsqueeze_1229), kwargs = {})
#   %add_509 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_622, %unsqueeze_1231), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 368640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 720
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


# kernel path: /tmp/torchinductor_sahanp/va/cvazuqfne3k4fd4zl54buyrxzfjz7qla4smbpk7ynpgf6h73iydk.py
# Topologically Sorted Source Nodes: [x_485, x_se_116], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_485 => add_510, clamp_max_161, clamp_min_161, div_161, mul_623
#   x_se_116 => mean_30
# Graph fragment:
#   %add_510 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_509, 3), kwargs = {})
#   %clamp_min_161 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_510, 0), kwargs = {})
#   %clamp_max_161 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_161, 6), kwargs = {})
#   %mul_623 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_509, %clamp_max_161), kwargs = {})
#   %div_161 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_623, 6), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_161, [2, 3], True), kwargs = {})
triton_per_fused_hardswish_mean_37 = async_compile.triton('triton_per_fused_hardswish_mean_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_37(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5760
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 720
    x1 = (xindex // 720)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (720*r2) + (46080*x1)), xmask, other=0.0)
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
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = 64.0
    tmp15 = tmp13 / tmp14
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vz/cvzvwotsizhis4yuls33epxk3wk6mfl4ej5grlzibzs44oolcjqt.py
# Topologically Sorted Source Nodes: [x_485, x_se_116, x_se_117, x_se_118, x_se_119, hardsigmoid_29, x_486], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_29 => add_512, clamp_max_163, clamp_min_163, div_163
#   x_485 => add_510, clamp_max_161, clamp_min_161, div_161, mul_623
#   x_486 => mul_625
#   x_se_116 => mean_30
#   x_se_117 => convolution_213
#   x_se_118 => add_511, clamp_max_162, clamp_min_162, div_162, mul_624
#   x_se_119 => convolution_214
# Graph fragment:
#   %add_510 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_509, 3), kwargs = {})
#   %clamp_min_161 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_510, 0), kwargs = {})
#   %clamp_max_161 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_161, 6), kwargs = {})
#   %mul_623 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_509, %clamp_max_161), kwargs = {})
#   %div_161 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_623, 6), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_161, [2, 3], True), kwargs = {})
#   %convolution_213 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_30, %arg380_1, %arg381_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_511 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_213, 3), kwargs = {})
#   %clamp_min_162 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_511, 0), kwargs = {})
#   %clamp_max_162 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_162, 6), kwargs = {})
#   %mul_624 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_213, %clamp_max_162), kwargs = {})
#   %div_162 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_624, 6), kwargs = {})
#   %convolution_214 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_162, %arg382_1, %arg383_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_512 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_214, 3), kwargs = {})
#   %clamp_min_163 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_512, 0), kwargs = {})
#   %clamp_max_163 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_163, 6), kwargs = {})
#   %div_163 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_163, 6), kwargs = {})
#   %mul_625 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_161, %div_163), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_38 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_38(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 368640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 720
    x2 = (xindex // 46080)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr0 + (x0 + (720*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ec/cec4yh2lavnrr2fms4r7tfjcn6ibwpiwdakks65sunt6cucfogym.py
# Topologically Sorted Source Nodes: [x_488], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_488 => add_514, mul_627, mul_628, sub_154
# Graph fragment:
#   %sub_154 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_215, %unsqueeze_1233), kwargs = {})
#   %mul_627 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_154, %unsqueeze_1235), kwargs = {})
#   %mul_628 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_627, %unsqueeze_1237), kwargs = {})
#   %add_514 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_628, %unsqueeze_1239), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 94208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 184
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


# kernel path: /tmp/torchinductor_sahanp/fd/cfd4kn4es3uozw5csgskaz6muugxlbd3rpwuuex5d6lyx5fbgizc.py
# Topologically Sorted Source Nodes: [x_490, x_491], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_490 => add_516, mul_630, mul_631, sub_155
#   x_491 => add_517, clamp_max_164, clamp_min_164, div_164, mul_632
# Graph fragment:
#   %sub_155 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_216, %unsqueeze_1241), kwargs = {})
#   %mul_630 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_155, %unsqueeze_1243), kwargs = {})
#   %mul_631 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_630, %unsqueeze_1245), kwargs = {})
#   %add_516 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_631, %unsqueeze_1247), kwargs = {})
#   %add_517 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_516, 3), kwargs = {})
#   %clamp_min_164 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_517, 0), kwargs = {})
#   %clamp_max_164 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_164, 6), kwargs = {})
#   %mul_632 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_516, %clamp_max_164), kwargs = {})
#   %div_164 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_632, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 736
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


# kernel path: /tmp/torchinductor_sahanp/zd/czdjeref7noye6ava2of3ziu7r66h53pijpkf7kr7fgocpynpovf.py
# Topologically Sorted Source Nodes: [x_493], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_493 => add_519, mul_634, mul_635, sub_156
# Graph fragment:
#   %sub_156 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_217, %unsqueeze_1249), kwargs = {})
#   %mul_634 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_156, %unsqueeze_1251), kwargs = {})
#   %mul_635 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_634, %unsqueeze_1253), kwargs = {})
#   %add_519 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_635, %unsqueeze_1255), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 736
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


# kernel path: /tmp/torchinductor_sahanp/jk/cjkzfbfheezgmlnpyfbom3zm6xy4jh6z2aygjneeo2vs5tym3tpg.py
# Topologically Sorted Source Nodes: [x_494, x_se_120], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_494 => add_520, clamp_max_165, clamp_min_165, div_165, mul_636
#   x_se_120 => mean_31
# Graph fragment:
#   %add_520 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_519, 3), kwargs = {})
#   %clamp_min_165 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_520, 0), kwargs = {})
#   %clamp_max_165 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_165, 6), kwargs = {})
#   %mul_636 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_519, %clamp_max_165), kwargs = {})
#   %div_165 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_636, 6), kwargs = {})
#   %mean_31 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_165, [2, 3], True), kwargs = {})
triton_per_fused_hardswish_mean_42 = async_compile.triton('triton_per_fused_hardswish_mean_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_42(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5888
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 736
    x1 = (xindex // 736)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (736*r2) + (47104*x1)), xmask, other=0.0)
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
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = 64.0
    tmp15 = tmp13 / tmp14
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3q/c3qgtlnxdiybzq4kuemdpbdfvfele2qxv2jzfjaxvekjldhu6n42.py
# Topologically Sorted Source Nodes: [x_494, x_se_120, x_se_121, x_se_122], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_494 => add_520, clamp_max_165, clamp_min_165, div_165, mul_636
#   x_se_120 => mean_31
#   x_se_121 => convolution_218
#   x_se_122 => add_521, clamp_max_166, clamp_min_166, div_166, mul_637
# Graph fragment:
#   %add_520 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_519, 3), kwargs = {})
#   %clamp_min_165 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_520, 0), kwargs = {})
#   %clamp_max_165 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_165, 6), kwargs = {})
#   %mul_636 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_519, %clamp_max_165), kwargs = {})
#   %div_165 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_636, 6), kwargs = {})
#   %mean_31 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_165, [2, 3], True), kwargs = {})
#   %convolution_218 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_31, %arg399_1, %arg400_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_521 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_218, 3), kwargs = {})
#   %clamp_min_166 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_521, 0), kwargs = {})
#   %clamp_max_166 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_166, 6), kwargs = {})
#   %mul_637 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_218, %clamp_max_166), kwargs = {})
#   %div_166 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_637, 6), kwargs = {})
triton_poi_fused_convolution_hardswish_mean_43 = async_compile.triton('triton_poi_fused_convolution_hardswish_mean_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardswish_mean_43(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
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


# kernel path: /tmp/torchinductor_sahanp/oo/cooki6f6qdfrrb2slmtar6ubzgdezxpsfxz6nrfkt56qgcyyual4.py
# Topologically Sorted Source Nodes: [x_494, x_se_120, x_se_121, x_se_122, x_se_123, hardsigmoid_30, x_495], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_30 => add_522, clamp_max_167, clamp_min_167, div_167
#   x_494 => add_520, clamp_max_165, clamp_min_165, div_165, mul_636
#   x_495 => mul_638
#   x_se_120 => mean_31
#   x_se_121 => convolution_218
#   x_se_122 => add_521, clamp_max_166, clamp_min_166, div_166, mul_637
#   x_se_123 => convolution_219
# Graph fragment:
#   %add_520 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_519, 3), kwargs = {})
#   %clamp_min_165 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_520, 0), kwargs = {})
#   %clamp_max_165 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_165, 6), kwargs = {})
#   %mul_636 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_519, %clamp_max_165), kwargs = {})
#   %div_165 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_636, 6), kwargs = {})
#   %mean_31 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_165, [2, 3], True), kwargs = {})
#   %convolution_218 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_31, %arg399_1, %arg400_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_521 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_218, 3), kwargs = {})
#   %clamp_min_166 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_521, 0), kwargs = {})
#   %clamp_max_166 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_166, 6), kwargs = {})
#   %mul_637 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_218, %clamp_max_166), kwargs = {})
#   %div_166 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_637, 6), kwargs = {})
#   %convolution_219 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_166, %arg401_1, %arg402_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_522 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_219, 3), kwargs = {})
#   %clamp_min_167 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_522, 0), kwargs = {})
#   %clamp_max_167 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_167, 6), kwargs = {})
#   %div_167 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_167, 6), kwargs = {})
#   %mul_638 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_165, %div_167), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_44 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_44(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 736
    x2 = (xindex // 47104)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr0 + (x0 + (736*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nc/cnc6yaoe6222tmg5lm6xa5bnve4oztby3nwhxcrygwe77lz7rd4v.py
# Topologically Sorted Source Nodes: [x_497, x_498], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_497 => add_524, mul_640, mul_641, sub_157
#   x_498 => add_525
# Graph fragment:
#   %sub_157 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_220, %unsqueeze_1257), kwargs = {})
#   %mul_640 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_157, %unsqueeze_1259), kwargs = {})
#   %mul_641 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_640, %unsqueeze_1261), kwargs = {})
#   %add_524 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_641, %unsqueeze_1263), kwargs = {})
#   %add_525 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_524, %add_514), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 94208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 184
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


# kernel path: /tmp/torchinductor_sahanp/77/c77ukgqsmxjz4jsf2b5f22u3tbg64unjmgmhlic3frcedxkffq3k.py
# Topologically Sorted Source Nodes: [x_540, x_541], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   x_540 => add_571, mul_695, mul_696, sub_170
#   x_541 => add_572, clamp_max_184, clamp_min_184, div_184, mul_697
# Graph fragment:
#   %sub_170 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_241, %unsqueeze_1361), kwargs = {})
#   %mul_695 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_170, %unsqueeze_1363), kwargs = {})
#   %mul_696 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_695, %unsqueeze_1365), kwargs = {})
#   %add_571 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_696, %unsqueeze_1367), kwargs = {})
#   %add_572 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_571, 3), kwargs = {})
#   %clamp_min_184 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_572, 0), kwargs = {})
#   %clamp_max_184 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_184, 6), kwargs = {})
#   %mul_697 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_571, %clamp_max_184), kwargs = {})
#   %div_184 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_697, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 565248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1104
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


# kernel path: /tmp/torchinductor_sahanp/cu/ccu7peq77no3wjtmkyavw2ebdze4jw3x7zksywoio26nwe4hd5il.py
# Topologically Sorted Source Nodes: [x_543], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_543 => add_574, mul_699, mul_700, sub_171
# Graph fragment:
#   %sub_171 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_242, %unsqueeze_1369), kwargs = {})
#   %mul_699 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_171, %unsqueeze_1371), kwargs = {})
#   %mul_700 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_699, %unsqueeze_1373), kwargs = {})
#   %add_574 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_700, %unsqueeze_1375), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 565248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1104
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


# kernel path: /tmp/torchinductor_sahanp/zt/cztvpkd6osbjtbabub3imgbzy7zq2zcdkdshjmtl725jpihylfiz.py
# Topologically Sorted Source Nodes: [x_544, x_se_140], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_544 => add_575, clamp_max_185, clamp_min_185, div_185, mul_701
#   x_se_140 => mean_36
# Graph fragment:
#   %add_575 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_574, 3), kwargs = {})
#   %clamp_min_185 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_575, 0), kwargs = {})
#   %clamp_max_185 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_185, 6), kwargs = {})
#   %mul_701 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_574, %clamp_max_185), kwargs = {})
#   %div_185 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_701, 6), kwargs = {})
#   %mean_36 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_185, [2, 3], True), kwargs = {})
triton_per_fused_hardswish_mean_48 = async_compile.triton('triton_per_fused_hardswish_mean_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_48(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8832
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 1104
    x1 = (xindex // 1104)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1104*r2) + (70656*x1)), xmask, other=0.0)
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
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = 64.0
    tmp15 = tmp13 / tmp14
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cl/cclfg5kparrv3dwn7aex2ubomdlsj4vehfb3hufodyrp7fu2c5rx.py
# Topologically Sorted Source Nodes: [x_544, x_se_140, x_se_141, x_se_142, x_se_143, hardsigmoid_35, x_545], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_35 => add_577, clamp_max_187, clamp_min_187, div_187
#   x_544 => add_575, clamp_max_185, clamp_min_185, div_185, mul_701
#   x_545 => mul_703
#   x_se_140 => mean_36
#   x_se_141 => convolution_243
#   x_se_142 => add_576, clamp_max_186, clamp_min_186, div_186, mul_702
#   x_se_143 => convolution_244
# Graph fragment:
#   %add_575 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_574, 3), kwargs = {})
#   %clamp_min_185 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_575, 0), kwargs = {})
#   %clamp_max_185 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_185, 6), kwargs = {})
#   %mul_701 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_574, %clamp_max_185), kwargs = {})
#   %div_185 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_701, 6), kwargs = {})
#   %mean_36 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_185, [2, 3], True), kwargs = {})
#   %convolution_243 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_36, %arg494_1, %arg495_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_576 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_243, 3), kwargs = {})
#   %clamp_min_186 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_576, 0), kwargs = {})
#   %clamp_max_186 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_186, 6), kwargs = {})
#   %mul_702 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_243, %clamp_max_186), kwargs = {})
#   %div_186 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_702, 6), kwargs = {})
#   %convolution_244 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_186, %arg496_1, %arg497_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_577 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_244, 3), kwargs = {})
#   %clamp_min_187 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_577, 0), kwargs = {})
#   %clamp_max_187 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_187, 6), kwargs = {})
#   %div_187 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_187, 6), kwargs = {})
#   %mul_703 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_185, %div_187), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_49 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_49(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 565248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 1104
    x2 = (xindex // 70656)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr0 + (x0 + (1104*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2c/c2c66mmbem7a6uodkz4oeslzlqljcmrswdz2crjwmzk7begje5rt.py
# Topologically Sorted Source Nodes: [x_547], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_547 => add_579, mul_705, mul_706, sub_172
# Graph fragment:
#   %sub_172 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_245, %unsqueeze_1377), kwargs = {})
#   %mul_705 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_172, %unsqueeze_1379), kwargs = {})
#   %mul_706 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_705, %unsqueeze_1381), kwargs = {})
#   %add_579 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_706, %unsqueeze_1383), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 114688
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dz/cdz7rzrorn7wxlve2hg7b3vonjo7kefyaak4apuc5yvclhtpyh3z.py
# Topologically Sorted Source Nodes: [x_549], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_549 => add_581, mul_708, mul_709, sub_173
# Graph fragment:
#   %sub_173 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_246, %unsqueeze_1385), kwargs = {})
#   %mul_708 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_173, %unsqueeze_1387), kwargs = {})
#   %mul_709 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_708, %unsqueeze_1389), kwargs = {})
#   %add_581 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_709, %unsqueeze_1391), kwargs = {})
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
    xnumel = 688128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1344
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


# kernel path: /tmp/torchinductor_sahanp/lp/clpui77mpxjmb3qbqqhvyvomind56owrclaxronr6o2dsc5tzkcd.py
# Topologically Sorted Source Nodes: [x_550, x_551], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   x_550 => add_582, clamp_max_188, clamp_min_188, div_188, mul_710
#   x_551 => mean_37
# Graph fragment:
#   %add_582 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_581, 3), kwargs = {})
#   %clamp_min_188 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_582, 0), kwargs = {})
#   %clamp_max_188 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_188, 6), kwargs = {})
#   %mul_710 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_581, %clamp_max_188), kwargs = {})
#   %div_188 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_710, 6), kwargs = {})
#   %mean_37 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div_188, [-1, -2], True), kwargs = {})
triton_per_fused_hardswish_mean_52 = async_compile.triton('triton_per_fused_hardswish_mean_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_52(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 1344
    x1 = (xindex // 1344)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1344*r2) + (86016*x1)), xmask, other=0.0)
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
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = 64.0
    tmp15 = tmp13 / tmp14
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rs/crszt7cld5nid5ugen6qv6bcywg4yhektstqkzu7p4qmmqsdntmr.py
# Topologically Sorted Source Nodes: [x_553], Original ATen: [aten.hardswish]
# Source node to ATen node mapping:
#   x_553 => add_583, clamp_max_189, clamp_min_189, div_189, mul_711
# Graph fragment:
#   %add_583 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_247, 3), kwargs = {})
#   %clamp_min_189 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_583, 0), kwargs = {})
#   %clamp_max_189 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_189, 6), kwargs = {})
#   %mul_711 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_247, %clamp_max_189), kwargs = {})
#   %div_189 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_711, 6), kwargs = {})
triton_poi_fused_hardswish_53 = async_compile.triton('triton_poi_fused_hardswish_53', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardswish_53(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
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
    assert_size_stride(arg16_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg17_1, (16, ), (1, ))
    assert_size_stride(arg18_1, (16, ), (1, ))
    assert_size_stride(arg19_1, (16, ), (1, ))
    assert_size_stride(arg20_1, (16, ), (1, ))
    assert_size_stride(arg21_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg22_1, (16, ), (1, ))
    assert_size_stride(arg23_1, (16, ), (1, ))
    assert_size_stride(arg24_1, (16, ), (1, ))
    assert_size_stride(arg25_1, (16, ), (1, ))
    assert_size_stride(arg26_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg27_1, (64, ), (1, ))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, ), (1, ))
    assert_size_stride(arg31_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg32_1, (64, ), (1, ))
    assert_size_stride(arg33_1, (64, ), (1, ))
    assert_size_stride(arg34_1, (64, ), (1, ))
    assert_size_stride(arg35_1, (64, ), (1, ))
    assert_size_stride(arg36_1, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg37_1, (24, ), (1, ))
    assert_size_stride(arg38_1, (24, ), (1, ))
    assert_size_stride(arg39_1, (24, ), (1, ))
    assert_size_stride(arg40_1, (24, ), (1, ))
    assert_size_stride(arg41_1, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg42_1, (48, ), (1, ))
    assert_size_stride(arg43_1, (48, ), (1, ))
    assert_size_stride(arg44_1, (48, ), (1, ))
    assert_size_stride(arg45_1, (48, ), (1, ))
    assert_size_stride(arg46_1, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg47_1, (48, ), (1, ))
    assert_size_stride(arg48_1, (48, ), (1, ))
    assert_size_stride(arg49_1, (48, ), (1, ))
    assert_size_stride(arg50_1, (48, ), (1, ))
    assert_size_stride(arg51_1, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg52_1, (24, ), (1, ))
    assert_size_stride(arg53_1, (24, ), (1, ))
    assert_size_stride(arg54_1, (24, ), (1, ))
    assert_size_stride(arg55_1, (24, ), (1, ))
    assert_size_stride(arg56_1, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg57_1, (48, ), (1, ))
    assert_size_stride(arg58_1, (48, ), (1, ))
    assert_size_stride(arg59_1, (48, ), (1, ))
    assert_size_stride(arg60_1, (48, ), (1, ))
    assert_size_stride(arg61_1, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg62_1, (48, ), (1, ))
    assert_size_stride(arg63_1, (48, ), (1, ))
    assert_size_stride(arg64_1, (48, ), (1, ))
    assert_size_stride(arg65_1, (48, ), (1, ))
    assert_size_stride(arg66_1, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg67_1, (24, ), (1, ))
    assert_size_stride(arg68_1, (24, ), (1, ))
    assert_size_stride(arg69_1, (24, ), (1, ))
    assert_size_stride(arg70_1, (24, ), (1, ))
    assert_size_stride(arg71_1, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg72_1, (48, ), (1, ))
    assert_size_stride(arg73_1, (48, ), (1, ))
    assert_size_stride(arg74_1, (48, ), (1, ))
    assert_size_stride(arg75_1, (48, ), (1, ))
    assert_size_stride(arg76_1, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg77_1, (48, ), (1, ))
    assert_size_stride(arg78_1, (48, ), (1, ))
    assert_size_stride(arg79_1, (48, ), (1, ))
    assert_size_stride(arg80_1, (48, ), (1, ))
    assert_size_stride(arg81_1, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg82_1, (24, ), (1, ))
    assert_size_stride(arg83_1, (24, ), (1, ))
    assert_size_stride(arg84_1, (24, ), (1, ))
    assert_size_stride(arg85_1, (24, ), (1, ))
    assert_size_stride(arg86_1, (120, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg87_1, (120, ), (1, ))
    assert_size_stride(arg88_1, (120, ), (1, ))
    assert_size_stride(arg89_1, (120, ), (1, ))
    assert_size_stride(arg90_1, (120, ), (1, ))
    assert_size_stride(arg91_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg92_1, (120, ), (1, ))
    assert_size_stride(arg93_1, (120, ), (1, ))
    assert_size_stride(arg94_1, (120, ), (1, ))
    assert_size_stride(arg95_1, (120, ), (1, ))
    assert_size_stride(arg96_1, (8, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg97_1, (8, ), (1, ))
    assert_size_stride(arg98_1, (120, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg99_1, (120, ), (1, ))
    assert_size_stride(arg100_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg101_1, (40, ), (1, ))
    assert_size_stride(arg102_1, (40, ), (1, ))
    assert_size_stride(arg103_1, (40, ), (1, ))
    assert_size_stride(arg104_1, (40, ), (1, ))
    assert_size_stride(arg105_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg106_1, (120, ), (1, ))
    assert_size_stride(arg107_1, (120, ), (1, ))
    assert_size_stride(arg108_1, (120, ), (1, ))
    assert_size_stride(arg109_1, (120, ), (1, ))
    assert_size_stride(arg110_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg111_1, (120, ), (1, ))
    assert_size_stride(arg112_1, (120, ), (1, ))
    assert_size_stride(arg113_1, (120, ), (1, ))
    assert_size_stride(arg114_1, (120, ), (1, ))
    assert_size_stride(arg115_1, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg116_1, (16, ), (1, ))
    assert_size_stride(arg117_1, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg118_1, (120, ), (1, ))
    assert_size_stride(arg119_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg120_1, (40, ), (1, ))
    assert_size_stride(arg121_1, (40, ), (1, ))
    assert_size_stride(arg122_1, (40, ), (1, ))
    assert_size_stride(arg123_1, (40, ), (1, ))
    assert_size_stride(arg124_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg125_1, (120, ), (1, ))
    assert_size_stride(arg126_1, (120, ), (1, ))
    assert_size_stride(arg127_1, (120, ), (1, ))
    assert_size_stride(arg128_1, (120, ), (1, ))
    assert_size_stride(arg129_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg130_1, (120, ), (1, ))
    assert_size_stride(arg131_1, (120, ), (1, ))
    assert_size_stride(arg132_1, (120, ), (1, ))
    assert_size_stride(arg133_1, (120, ), (1, ))
    assert_size_stride(arg134_1, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg135_1, (16, ), (1, ))
    assert_size_stride(arg136_1, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg137_1, (120, ), (1, ))
    assert_size_stride(arg138_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg139_1, (40, ), (1, ))
    assert_size_stride(arg140_1, (40, ), (1, ))
    assert_size_stride(arg141_1, (40, ), (1, ))
    assert_size_stride(arg142_1, (40, ), (1, ))
    assert_size_stride(arg143_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg144_1, (120, ), (1, ))
    assert_size_stride(arg145_1, (120, ), (1, ))
    assert_size_stride(arg146_1, (120, ), (1, ))
    assert_size_stride(arg147_1, (120, ), (1, ))
    assert_size_stride(arg148_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg149_1, (120, ), (1, ))
    assert_size_stride(arg150_1, (120, ), (1, ))
    assert_size_stride(arg151_1, (120, ), (1, ))
    assert_size_stride(arg152_1, (120, ), (1, ))
    assert_size_stride(arg153_1, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg154_1, (16, ), (1, ))
    assert_size_stride(arg155_1, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg156_1, (120, ), (1, ))
    assert_size_stride(arg157_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg158_1, (40, ), (1, ))
    assert_size_stride(arg159_1, (40, ), (1, ))
    assert_size_stride(arg160_1, (40, ), (1, ))
    assert_size_stride(arg161_1, (40, ), (1, ))
    assert_size_stride(arg162_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg163_1, (120, ), (1, ))
    assert_size_stride(arg164_1, (120, ), (1, ))
    assert_size_stride(arg165_1, (120, ), (1, ))
    assert_size_stride(arg166_1, (120, ), (1, ))
    assert_size_stride(arg167_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg168_1, (120, ), (1, ))
    assert_size_stride(arg169_1, (120, ), (1, ))
    assert_size_stride(arg170_1, (120, ), (1, ))
    assert_size_stride(arg171_1, (120, ), (1, ))
    assert_size_stride(arg172_1, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg173_1, (16, ), (1, ))
    assert_size_stride(arg174_1, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg175_1, (120, ), (1, ))
    assert_size_stride(arg176_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg177_1, (40, ), (1, ))
    assert_size_stride(arg178_1, (40, ), (1, ))
    assert_size_stride(arg179_1, (40, ), (1, ))
    assert_size_stride(arg180_1, (40, ), (1, ))
    assert_size_stride(arg181_1, (200, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg182_1, (200, ), (1, ))
    assert_size_stride(arg183_1, (200, ), (1, ))
    assert_size_stride(arg184_1, (200, ), (1, ))
    assert_size_stride(arg185_1, (200, ), (1, ))
    assert_size_stride(arg186_1, (200, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg187_1, (200, ), (1, ))
    assert_size_stride(arg188_1, (200, ), (1, ))
    assert_size_stride(arg189_1, (200, ), (1, ))
    assert_size_stride(arg190_1, (200, ), (1, ))
    assert_size_stride(arg191_1, (72, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg192_1, (72, ), (1, ))
    assert_size_stride(arg193_1, (72, ), (1, ))
    assert_size_stride(arg194_1, (72, ), (1, ))
    assert_size_stride(arg195_1, (72, ), (1, ))
    assert_size_stride(arg196_1, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg197_1, (216, ), (1, ))
    assert_size_stride(arg198_1, (216, ), (1, ))
    assert_size_stride(arg199_1, (216, ), (1, ))
    assert_size_stride(arg200_1, (216, ), (1, ))
    assert_size_stride(arg201_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg202_1, (216, ), (1, ))
    assert_size_stride(arg203_1, (216, ), (1, ))
    assert_size_stride(arg204_1, (216, ), (1, ))
    assert_size_stride(arg205_1, (216, ), (1, ))
    assert_size_stride(arg206_1, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg207_1, (72, ), (1, ))
    assert_size_stride(arg208_1, (72, ), (1, ))
    assert_size_stride(arg209_1, (72, ), (1, ))
    assert_size_stride(arg210_1, (72, ), (1, ))
    assert_size_stride(arg211_1, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg212_1, (216, ), (1, ))
    assert_size_stride(arg213_1, (216, ), (1, ))
    assert_size_stride(arg214_1, (216, ), (1, ))
    assert_size_stride(arg215_1, (216, ), (1, ))
    assert_size_stride(arg216_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg217_1, (216, ), (1, ))
    assert_size_stride(arg218_1, (216, ), (1, ))
    assert_size_stride(arg219_1, (216, ), (1, ))
    assert_size_stride(arg220_1, (216, ), (1, ))
    assert_size_stride(arg221_1, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg222_1, (72, ), (1, ))
    assert_size_stride(arg223_1, (72, ), (1, ))
    assert_size_stride(arg224_1, (72, ), (1, ))
    assert_size_stride(arg225_1, (72, ), (1, ))
    assert_size_stride(arg226_1, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg227_1, (216, ), (1, ))
    assert_size_stride(arg228_1, (216, ), (1, ))
    assert_size_stride(arg229_1, (216, ), (1, ))
    assert_size_stride(arg230_1, (216, ), (1, ))
    assert_size_stride(arg231_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg232_1, (216, ), (1, ))
    assert_size_stride(arg233_1, (216, ), (1, ))
    assert_size_stride(arg234_1, (216, ), (1, ))
    assert_size_stride(arg235_1, (216, ), (1, ))
    assert_size_stride(arg236_1, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg237_1, (72, ), (1, ))
    assert_size_stride(arg238_1, (72, ), (1, ))
    assert_size_stride(arg239_1, (72, ), (1, ))
    assert_size_stride(arg240_1, (72, ), (1, ))
    assert_size_stride(arg241_1, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg242_1, (216, ), (1, ))
    assert_size_stride(arg243_1, (216, ), (1, ))
    assert_size_stride(arg244_1, (216, ), (1, ))
    assert_size_stride(arg245_1, (216, ), (1, ))
    assert_size_stride(arg246_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg247_1, (216, ), (1, ))
    assert_size_stride(arg248_1, (216, ), (1, ))
    assert_size_stride(arg249_1, (216, ), (1, ))
    assert_size_stride(arg250_1, (216, ), (1, ))
    assert_size_stride(arg251_1, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg252_1, (72, ), (1, ))
    assert_size_stride(arg253_1, (72, ), (1, ))
    assert_size_stride(arg254_1, (72, ), (1, ))
    assert_size_stride(arg255_1, (72, ), (1, ))
    assert_size_stride(arg256_1, (360, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg257_1, (360, ), (1, ))
    assert_size_stride(arg258_1, (360, ), (1, ))
    assert_size_stride(arg259_1, (360, ), (1, ))
    assert_size_stride(arg260_1, (360, ), (1, ))
    assert_size_stride(arg261_1, (360, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg262_1, (360, ), (1, ))
    assert_size_stride(arg263_1, (360, ), (1, ))
    assert_size_stride(arg264_1, (360, ), (1, ))
    assert_size_stride(arg265_1, (360, ), (1, ))
    assert_size_stride(arg266_1, (24, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg267_1, (24, ), (1, ))
    assert_size_stride(arg268_1, (360, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg269_1, (360, ), (1, ))
    assert_size_stride(arg270_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg271_1, (120, ), (1, ))
    assert_size_stride(arg272_1, (120, ), (1, ))
    assert_size_stride(arg273_1, (120, ), (1, ))
    assert_size_stride(arg274_1, (120, ), (1, ))
    assert_size_stride(arg275_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg276_1, (360, ), (1, ))
    assert_size_stride(arg277_1, (360, ), (1, ))
    assert_size_stride(arg278_1, (360, ), (1, ))
    assert_size_stride(arg279_1, (360, ), (1, ))
    assert_size_stride(arg280_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg281_1, (360, ), (1, ))
    assert_size_stride(arg282_1, (360, ), (1, ))
    assert_size_stride(arg283_1, (360, ), (1, ))
    assert_size_stride(arg284_1, (360, ), (1, ))
    assert_size_stride(arg285_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg286_1, (32, ), (1, ))
    assert_size_stride(arg287_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg288_1, (360, ), (1, ))
    assert_size_stride(arg289_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg290_1, (120, ), (1, ))
    assert_size_stride(arg291_1, (120, ), (1, ))
    assert_size_stride(arg292_1, (120, ), (1, ))
    assert_size_stride(arg293_1, (120, ), (1, ))
    assert_size_stride(arg294_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg295_1, (360, ), (1, ))
    assert_size_stride(arg296_1, (360, ), (1, ))
    assert_size_stride(arg297_1, (360, ), (1, ))
    assert_size_stride(arg298_1, (360, ), (1, ))
    assert_size_stride(arg299_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg300_1, (360, ), (1, ))
    assert_size_stride(arg301_1, (360, ), (1, ))
    assert_size_stride(arg302_1, (360, ), (1, ))
    assert_size_stride(arg303_1, (360, ), (1, ))
    assert_size_stride(arg304_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg305_1, (32, ), (1, ))
    assert_size_stride(arg306_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg307_1, (360, ), (1, ))
    assert_size_stride(arg308_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg309_1, (120, ), (1, ))
    assert_size_stride(arg310_1, (120, ), (1, ))
    assert_size_stride(arg311_1, (120, ), (1, ))
    assert_size_stride(arg312_1, (120, ), (1, ))
    assert_size_stride(arg313_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg314_1, (360, ), (1, ))
    assert_size_stride(arg315_1, (360, ), (1, ))
    assert_size_stride(arg316_1, (360, ), (1, ))
    assert_size_stride(arg317_1, (360, ), (1, ))
    assert_size_stride(arg318_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg319_1, (360, ), (1, ))
    assert_size_stride(arg320_1, (360, ), (1, ))
    assert_size_stride(arg321_1, (360, ), (1, ))
    assert_size_stride(arg322_1, (360, ), (1, ))
    assert_size_stride(arg323_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg324_1, (32, ), (1, ))
    assert_size_stride(arg325_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg326_1, (360, ), (1, ))
    assert_size_stride(arg327_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg328_1, (120, ), (1, ))
    assert_size_stride(arg329_1, (120, ), (1, ))
    assert_size_stride(arg330_1, (120, ), (1, ))
    assert_size_stride(arg331_1, (120, ), (1, ))
    assert_size_stride(arg332_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg333_1, (360, ), (1, ))
    assert_size_stride(arg334_1, (360, ), (1, ))
    assert_size_stride(arg335_1, (360, ), (1, ))
    assert_size_stride(arg336_1, (360, ), (1, ))
    assert_size_stride(arg337_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg338_1, (360, ), (1, ))
    assert_size_stride(arg339_1, (360, ), (1, ))
    assert_size_stride(arg340_1, (360, ), (1, ))
    assert_size_stride(arg341_1, (360, ), (1, ))
    assert_size_stride(arg342_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg343_1, (32, ), (1, ))
    assert_size_stride(arg344_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg345_1, (360, ), (1, ))
    assert_size_stride(arg346_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg347_1, (120, ), (1, ))
    assert_size_stride(arg348_1, (120, ), (1, ))
    assert_size_stride(arg349_1, (120, ), (1, ))
    assert_size_stride(arg350_1, (120, ), (1, ))
    assert_size_stride(arg351_1, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg352_1, (360, ), (1, ))
    assert_size_stride(arg353_1, (360, ), (1, ))
    assert_size_stride(arg354_1, (360, ), (1, ))
    assert_size_stride(arg355_1, (360, ), (1, ))
    assert_size_stride(arg356_1, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg357_1, (360, ), (1, ))
    assert_size_stride(arg358_1, (360, ), (1, ))
    assert_size_stride(arg359_1, (360, ), (1, ))
    assert_size_stride(arg360_1, (360, ), (1, ))
    assert_size_stride(arg361_1, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg362_1, (32, ), (1, ))
    assert_size_stride(arg363_1, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg364_1, (360, ), (1, ))
    assert_size_stride(arg365_1, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(arg366_1, (120, ), (1, ))
    assert_size_stride(arg367_1, (120, ), (1, ))
    assert_size_stride(arg368_1, (120, ), (1, ))
    assert_size_stride(arg369_1, (120, ), (1, ))
    assert_size_stride(arg370_1, (720, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg371_1, (720, ), (1, ))
    assert_size_stride(arg372_1, (720, ), (1, ))
    assert_size_stride(arg373_1, (720, ), (1, ))
    assert_size_stride(arg374_1, (720, ), (1, ))
    assert_size_stride(arg375_1, (720, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg376_1, (720, ), (1, ))
    assert_size_stride(arg377_1, (720, ), (1, ))
    assert_size_stride(arg378_1, (720, ), (1, ))
    assert_size_stride(arg379_1, (720, ), (1, ))
    assert_size_stride(arg380_1, (32, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(arg381_1, (32, ), (1, ))
    assert_size_stride(arg382_1, (720, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg383_1, (720, ), (1, ))
    assert_size_stride(arg384_1, (184, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(arg385_1, (184, ), (1, ))
    assert_size_stride(arg386_1, (184, ), (1, ))
    assert_size_stride(arg387_1, (184, ), (1, ))
    assert_size_stride(arg388_1, (184, ), (1, ))
    assert_size_stride(arg389_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg390_1, (736, ), (1, ))
    assert_size_stride(arg391_1, (736, ), (1, ))
    assert_size_stride(arg392_1, (736, ), (1, ))
    assert_size_stride(arg393_1, (736, ), (1, ))
    assert_size_stride(arg394_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg395_1, (736, ), (1, ))
    assert_size_stride(arg396_1, (736, ), (1, ))
    assert_size_stride(arg397_1, (736, ), (1, ))
    assert_size_stride(arg398_1, (736, ), (1, ))
    assert_size_stride(arg399_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg400_1, (48, ), (1, ))
    assert_size_stride(arg401_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg402_1, (736, ), (1, ))
    assert_size_stride(arg403_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg404_1, (184, ), (1, ))
    assert_size_stride(arg405_1, (184, ), (1, ))
    assert_size_stride(arg406_1, (184, ), (1, ))
    assert_size_stride(arg407_1, (184, ), (1, ))
    assert_size_stride(arg408_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg409_1, (736, ), (1, ))
    assert_size_stride(arg410_1, (736, ), (1, ))
    assert_size_stride(arg411_1, (736, ), (1, ))
    assert_size_stride(arg412_1, (736, ), (1, ))
    assert_size_stride(arg413_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg414_1, (736, ), (1, ))
    assert_size_stride(arg415_1, (736, ), (1, ))
    assert_size_stride(arg416_1, (736, ), (1, ))
    assert_size_stride(arg417_1, (736, ), (1, ))
    assert_size_stride(arg418_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg419_1, (48, ), (1, ))
    assert_size_stride(arg420_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg421_1, (736, ), (1, ))
    assert_size_stride(arg422_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg423_1, (184, ), (1, ))
    assert_size_stride(arg424_1, (184, ), (1, ))
    assert_size_stride(arg425_1, (184, ), (1, ))
    assert_size_stride(arg426_1, (184, ), (1, ))
    assert_size_stride(arg427_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg428_1, (736, ), (1, ))
    assert_size_stride(arg429_1, (736, ), (1, ))
    assert_size_stride(arg430_1, (736, ), (1, ))
    assert_size_stride(arg431_1, (736, ), (1, ))
    assert_size_stride(arg432_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg433_1, (736, ), (1, ))
    assert_size_stride(arg434_1, (736, ), (1, ))
    assert_size_stride(arg435_1, (736, ), (1, ))
    assert_size_stride(arg436_1, (736, ), (1, ))
    assert_size_stride(arg437_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg438_1, (48, ), (1, ))
    assert_size_stride(arg439_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg440_1, (736, ), (1, ))
    assert_size_stride(arg441_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg442_1, (184, ), (1, ))
    assert_size_stride(arg443_1, (184, ), (1, ))
    assert_size_stride(arg444_1, (184, ), (1, ))
    assert_size_stride(arg445_1, (184, ), (1, ))
    assert_size_stride(arg446_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg447_1, (736, ), (1, ))
    assert_size_stride(arg448_1, (736, ), (1, ))
    assert_size_stride(arg449_1, (736, ), (1, ))
    assert_size_stride(arg450_1, (736, ), (1, ))
    assert_size_stride(arg451_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg452_1, (736, ), (1, ))
    assert_size_stride(arg453_1, (736, ), (1, ))
    assert_size_stride(arg454_1, (736, ), (1, ))
    assert_size_stride(arg455_1, (736, ), (1, ))
    assert_size_stride(arg456_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg457_1, (48, ), (1, ))
    assert_size_stride(arg458_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg459_1, (736, ), (1, ))
    assert_size_stride(arg460_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg461_1, (184, ), (1, ))
    assert_size_stride(arg462_1, (184, ), (1, ))
    assert_size_stride(arg463_1, (184, ), (1, ))
    assert_size_stride(arg464_1, (184, ), (1, ))
    assert_size_stride(arg465_1, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg466_1, (736, ), (1, ))
    assert_size_stride(arg467_1, (736, ), (1, ))
    assert_size_stride(arg468_1, (736, ), (1, ))
    assert_size_stride(arg469_1, (736, ), (1, ))
    assert_size_stride(arg470_1, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg471_1, (736, ), (1, ))
    assert_size_stride(arg472_1, (736, ), (1, ))
    assert_size_stride(arg473_1, (736, ), (1, ))
    assert_size_stride(arg474_1, (736, ), (1, ))
    assert_size_stride(arg475_1, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg476_1, (48, ), (1, ))
    assert_size_stride(arg477_1, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg478_1, (736, ), (1, ))
    assert_size_stride(arg479_1, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg480_1, (184, ), (1, ))
    assert_size_stride(arg481_1, (184, ), (1, ))
    assert_size_stride(arg482_1, (184, ), (1, ))
    assert_size_stride(arg483_1, (184, ), (1, ))
    assert_size_stride(arg484_1, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg485_1, (1104, ), (1, ))
    assert_size_stride(arg486_1, (1104, ), (1, ))
    assert_size_stride(arg487_1, (1104, ), (1, ))
    assert_size_stride(arg488_1, (1104, ), (1, ))
    assert_size_stride(arg489_1, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg490_1, (1104, ), (1, ))
    assert_size_stride(arg491_1, (1104, ), (1, ))
    assert_size_stride(arg492_1, (1104, ), (1, ))
    assert_size_stride(arg493_1, (1104, ), (1, ))
    assert_size_stride(arg494_1, (48, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg495_1, (48, ), (1, ))
    assert_size_stride(arg496_1, (1104, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg497_1, (1104, ), (1, ))
    assert_size_stride(arg498_1, (224, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(arg499_1, (224, ), (1, ))
    assert_size_stride(arg500_1, (224, ), (1, ))
    assert_size_stride(arg501_1, (224, ), (1, ))
    assert_size_stride(arg502_1, (224, ), (1, ))
    assert_size_stride(arg503_1, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg504_1, (1344, ), (1, ))
    assert_size_stride(arg505_1, (1344, ), (1, ))
    assert_size_stride(arg506_1, (1344, ), (1, ))
    assert_size_stride(arg507_1, (1344, ), (1, ))
    assert_size_stride(arg508_1, (1984, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(arg509_1, (1000, 1984), (1984, 1))
    assert_size_stride(arg510_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 256, 256), (196608, 1, 768, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_278], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((16, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_278], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 48, 9, grid=grid(48, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_278], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 16, 128, 128), (262144, 1, 2048, 16))
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((8, 16, 128, 128), (262144, 1, 2048, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_279, x_280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, buf4, 2097152, grid=grid(2097152), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        # Topologically Sorted Source Nodes: [x_280, x_281], Original ATen: [aten.hardswish, aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf5, (8, 16, 128, 128), (262144, 1, 2048, 16))
        del arg6_1
        buf6 = buf5; del buf5  # reuse
        buf7 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_282, x_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, buf7, 2097152, grid=grid(2097152), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        del buf6
        # Topologically Sorted Source Nodes: [x_283, x_284], Original ATen: [aten.hardswish, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 16, 128, 128), (262144, 1, 2048, 16))
        del arg11_1
        del buf7
        buf9 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_285, x_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_3.run(buf9, buf8, arg12_1, arg13_1, arg14_1, arg15_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        # Topologically Sorted Source Nodes: [x_287], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg16_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf10, (8, 16, 128, 128), (262144, 1, 2048, 16))
        del arg16_1
        buf11 = buf10; del buf10  # reuse
        buf12 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_288, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf11, arg17_1, arg18_1, arg19_1, arg20_1, buf12, 2097152, grid=grid(2097152), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del buf11
        # Topologically Sorted Source Nodes: [x_289, x_290], Original ATen: [aten.hardswish, aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 16, 128, 128), (262144, 1, 2048, 16))
        del arg21_1
        del buf12
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_291, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_4.run(buf14, arg22_1, arg23_1, arg24_1, arg25_1, buf9, 2097152, grid=grid(2097152), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del buf9
        # Topologically Sorted Source Nodes: [x_291, x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 64, 128, 128), (1048576, 1, 8192, 64))
        del arg26_1
        buf16 = buf15; del buf15  # reuse
        buf17 = empty_strided_cuda((8, 64, 128, 128), (1048576, 1, 8192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_294, x_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5.run(buf16, arg27_1, arg28_1, arg29_1, arg30_1, buf17, 8388608, grid=grid(8388608), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        del buf16
        # Topologically Sorted Source Nodes: [x_295, x_296], Original ATen: [aten.hardswish, aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg31_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf18, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del arg31_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        buf20 = reinterpret_tensor(buf14, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_297, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6.run(buf19, arg32_1, arg33_1, arg34_1, arg35_1, buf20, 2097152, grid=grid(2097152), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        del buf19
        # Topologically Sorted Source Nodes: [x_298, x_299], Original ATen: [aten.hardswish, aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 24, 64, 64), (98304, 1, 1536, 24))
        del arg36_1
        del buf20
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_300], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_7.run(buf22, arg37_1, arg38_1, arg39_1, arg40_1, 786432, grid=grid(786432), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        # Topologically Sorted Source Nodes: [x_301], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 48, 64, 64), (196608, 1, 3072, 48))
        del arg41_1
        buf24 = buf23; del buf23  # reuse
        buf25 = reinterpret_tensor(buf0, (8, 48, 64, 64), (196608, 1, 3072, 48), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_302, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8.run(buf24, arg42_1, arg43_1, arg44_1, arg45_1, buf25, 1572864, grid=grid(1572864), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        del buf24
        # Topologically Sorted Source Nodes: [x_303, x_304], Original ATen: [aten.hardswish, aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg46_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf26, (8, 48, 64, 64), (196608, 1, 3072, 48))
        del arg46_1
        buf27 = buf26; del buf26  # reuse
        buf28 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_305, x_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8.run(buf27, arg47_1, arg48_1, arg49_1, arg50_1, buf28, 1572864, grid=grid(1572864), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        del buf27
        # Topologically Sorted Source Nodes: [x_306, x_307], Original ATen: [aten.hardswish, aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 24, 64, 64), (98304, 1, 1536, 24))
        del arg51_1
        buf30 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_308, x_309], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_9.run(buf30, buf29, arg52_1, arg53_1, arg54_1, arg55_1, 786432, grid=grid(786432), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        del buf29
        # Topologically Sorted Source Nodes: [x_310], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 48, 64, 64), (196608, 1, 3072, 48))
        del arg56_1
        buf32 = buf31; del buf31  # reuse
        buf33 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_311, x_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8.run(buf32, arg57_1, arg58_1, arg59_1, arg60_1, buf33, 1572864, grid=grid(1572864), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf32
        # Topologically Sorted Source Nodes: [x_312, x_313], Original ATen: [aten.hardswish, aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg61_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf34, (8, 48, 64, 64), (196608, 1, 3072, 48))
        del arg61_1
        buf35 = buf34; del buf34  # reuse
        buf36 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_314, x_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8.run(buf35, arg62_1, arg63_1, arg64_1, arg65_1, buf36, 1572864, grid=grid(1572864), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        del buf35
        # Topologically Sorted Source Nodes: [x_315, x_316], Original ATen: [aten.hardswish, aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 24, 64, 64), (98304, 1, 1536, 24))
        del arg66_1
        buf38 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_317, x_318], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_9.run(buf38, buf37, arg67_1, arg68_1, arg69_1, arg70_1, 786432, grid=grid(786432), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        del buf37
        # Topologically Sorted Source Nodes: [x_319], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 48, 64, 64), (196608, 1, 3072, 48))
        del arg71_1
        buf40 = buf39; del buf39  # reuse
        buf41 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_320, x_321], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8.run(buf40, arg72_1, arg73_1, arg74_1, arg75_1, buf41, 1572864, grid=grid(1572864), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        del buf40
        # Topologically Sorted Source Nodes: [x_321, x_322], Original ATen: [aten.hardswish, aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg76_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf42, (8, 48, 64, 64), (196608, 1, 3072, 48))
        del arg76_1
        buf43 = buf42; del buf42  # reuse
        buf44 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_323, x_324], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8.run(buf43, arg77_1, arg78_1, arg79_1, arg80_1, buf44, 1572864, grid=grid(1572864), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        del buf43
        # Topologically Sorted Source Nodes: [x_324, x_325], Original ATen: [aten.hardswish, aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 24, 64, 64), (98304, 1, 1536, 24))
        del arg81_1
        del buf44
        buf46 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_326, x_327], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_9.run(buf46, buf45, arg82_1, arg83_1, arg84_1, arg85_1, 786432, grid=grid(786432), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        del buf45
        # Topologically Sorted Source Nodes: [x_326, x_327, x_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 120, 64, 64), (491520, 1, 7680, 120))
        del arg86_1
        del buf46
        buf48 = buf47; del buf47  # reuse
        buf49 = empty_strided_cuda((8, 120, 64, 64), (491520, 1, 7680, 120), torch.float32)
        # Topologically Sorted Source Nodes: [x_329, x_330], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_10.run(buf48, arg87_1, arg88_1, arg89_1, arg90_1, buf49, 3932160, grid=grid(3932160), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        del buf48
        # Topologically Sorted Source Nodes: [x_330, x_331], Original ATen: [aten.hardswish, aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg91_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf50, (8, 120, 32, 32), (122880, 1, 3840, 120))
        del arg91_1
        del buf49
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_332], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf51, arg92_1, arg93_1, arg94_1, arg95_1, 983040, grid=grid(983040), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        buf52 = empty_strided_cuda((8, 120, 1, 1, 8), (960, 1, 7680, 7680, 120), torch.float32)
        # Topologically Sorted Source Nodes: [x_333, x_se_72], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_12.run(buf51, buf52, 7680, 128, grid=grid(7680), stream=stream0)
        buf54 = empty_strided_cuda((8, 120, 1, 1), (120, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_333, x_se_72], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_13.run(buf52, buf54, 960, 8, grid=grid(960), stream=stream0)
        # Topologically Sorted Source Nodes: [x_333, x_se_72, x_se_73], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf55 = extern_kernels.convolution(buf54, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg96_1
        del buf54
        buf56 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_333, x_se_72, x_se_73, x_se_74], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_14.run(buf56, arg97_1, 64, grid=grid(64), stream=stream0)
        del arg97_1
        # Topologically Sorted Source Nodes: [x_333, x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf57 = extern_kernels.convolution(buf56, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg98_1
        del buf56
        buf58 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_333, x_se_72, x_se_73, x_se_74, x_se_75, hardsigmoid_18, x_334], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_15.run(buf58, buf57, arg99_1, 983040, grid=grid(983040), stream=stream0)
        del arg99_1
        # Topologically Sorted Source Nodes: [x_333, x_se_72, x_se_73, x_se_74, x_se_75, hardsigmoid_18, x_334, x_335], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf59 = extern_kernels.convolution(buf58, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 40, 32, 32), (40960, 1, 1280, 40))
        del arg100_1
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_336], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf60, arg101_1, arg102_1, arg103_1, arg104_1, 327680, grid=grid(327680), stream=stream0)
        del arg101_1
        del arg102_1
        del arg103_1
        del arg104_1
        # Topologically Sorted Source Nodes: [x_337], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (8, 120, 32, 32), (122880, 1, 3840, 120))
        del arg105_1
        buf62 = buf61; del buf61  # reuse
        buf63 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_338, x_339], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_17.run(buf62, arg106_1, arg107_1, arg108_1, arg109_1, buf63, 983040, grid=grid(983040), stream=stream0)
        del arg106_1
        del arg107_1
        del arg108_1
        del arg109_1
        del buf62
        # Topologically Sorted Source Nodes: [x_339, x_340], Original ATen: [aten.hardswish, aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg110_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf64, (8, 120, 32, 32), (122880, 1, 3840, 120))
        del arg110_1
        del buf63
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_341], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf65, arg111_1, arg112_1, arg113_1, arg114_1, 983040, grid=grid(983040), stream=stream0)
        del arg111_1
        del arg112_1
        del arg113_1
        del arg114_1
        buf66 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_342, x_se_76], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_12.run(buf65, buf66, 7680, 128, grid=grid(7680), stream=stream0)
        buf68 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_342, x_se_76], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_13.run(buf66, buf68, 960, 8, grid=grid(960), stream=stream0)
        # Topologically Sorted Source Nodes: [x_342, x_se_76, x_se_77], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf69 = extern_kernels.convolution(buf68, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 16, 1, 1), (16, 1, 1, 1))
        del arg115_1
        del buf68
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_342, x_se_76, x_se_77, x_se_78], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_18.run(buf70, arg116_1, 128, grid=grid(128), stream=stream0)
        del arg116_1
        # Topologically Sorted Source Nodes: [x_342, x_se_76, x_se_77, x_se_78, x_se_79], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf71 = extern_kernels.convolution(buf70, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg117_1
        del buf70
        buf72 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_342, x_se_76, x_se_77, x_se_78, x_se_79, hardsigmoid_19, x_343], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_15.run(buf72, buf71, arg118_1, 983040, grid=grid(983040), stream=stream0)
        del arg118_1
        # Topologically Sorted Source Nodes: [x_342, x_se_76, x_se_77, x_se_78, x_se_79, hardsigmoid_19, x_343, x_344], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf73 = extern_kernels.convolution(buf72, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 40, 32, 32), (40960, 1, 1280, 40))
        del arg119_1
        buf74 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_345, x_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_19.run(buf74, buf73, arg120_1, arg121_1, arg122_1, arg123_1, 327680, grid=grid(327680), stream=stream0)
        del arg120_1
        del arg121_1
        del arg122_1
        del arg123_1
        del buf73
        # Topologically Sorted Source Nodes: [x_347], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg124_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 120, 32, 32), (122880, 1, 3840, 120))
        del arg124_1
        buf76 = buf75; del buf75  # reuse
        buf77 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_348, x_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_17.run(buf76, arg125_1, arg126_1, arg127_1, arg128_1, buf77, 983040, grid=grid(983040), stream=stream0)
        del arg125_1
        del arg126_1
        del arg127_1
        del arg128_1
        del buf76
        # Topologically Sorted Source Nodes: [x_349, x_350], Original ATen: [aten.hardswish, aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg129_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf78, (8, 120, 32, 32), (122880, 1, 3840, 120))
        del arg129_1
        del buf77
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_351], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf79, arg130_1, arg131_1, arg132_1, arg133_1, 983040, grid=grid(983040), stream=stream0)
        del arg130_1
        del arg131_1
        del arg132_1
        del arg133_1
        buf80 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_352, x_se_80], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_12.run(buf79, buf80, 7680, 128, grid=grid(7680), stream=stream0)
        buf82 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_352, x_se_80], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_13.run(buf80, buf82, 960, 8, grid=grid(960), stream=stream0)
        # Topologically Sorted Source Nodes: [x_352, x_se_80, x_se_81], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf83 = extern_kernels.convolution(buf82, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (8, 16, 1, 1), (16, 1, 1, 1))
        del arg134_1
        del buf82
        buf84 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_352, x_se_80, x_se_81, x_se_82], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_18.run(buf84, arg135_1, 128, grid=grid(128), stream=stream0)
        del arg135_1
        # Topologically Sorted Source Nodes: [x_352, x_se_80, x_se_81, x_se_82, x_se_83], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf85 = extern_kernels.convolution(buf84, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg136_1
        del buf84
        buf86 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [x_352, x_se_80, x_se_81, x_se_82, x_se_83, hardsigmoid_20, x_353], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_15.run(buf86, buf85, arg137_1, 983040, grid=grid(983040), stream=stream0)
        del arg137_1
        # Topologically Sorted Source Nodes: [x_352, x_se_80, x_se_81, x_se_82, x_se_83, hardsigmoid_20, x_353, x_354], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf87 = extern_kernels.convolution(buf86, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 40, 32, 32), (40960, 1, 1280, 40))
        del arg138_1
        buf88 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_355, x_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_19.run(buf88, buf87, arg139_1, arg140_1, arg141_1, arg142_1, 327680, grid=grid(327680), stream=stream0)
        del arg139_1
        del arg140_1
        del arg141_1
        del arg142_1
        del buf87
        # Topologically Sorted Source Nodes: [x_357], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 120, 32, 32), (122880, 1, 3840, 120))
        del arg143_1
        buf90 = buf89; del buf89  # reuse
        buf91 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_358, x_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_17.run(buf90, arg144_1, arg145_1, arg146_1, arg147_1, buf91, 983040, grid=grid(983040), stream=stream0)
        del arg144_1
        del arg145_1
        del arg146_1
        del arg147_1
        del buf90
        # Topologically Sorted Source Nodes: [x_359, x_360], Original ATen: [aten.hardswish, aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg148_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf92, (8, 120, 32, 32), (122880, 1, 3840, 120))
        del arg148_1
        del buf91
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_361], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf93, arg149_1, arg150_1, arg151_1, arg152_1, 983040, grid=grid(983040), stream=stream0)
        del arg149_1
        del arg150_1
        del arg151_1
        del arg152_1
        buf94 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_362, x_se_84], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_12.run(buf93, buf94, 7680, 128, grid=grid(7680), stream=stream0)
        buf96 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_362, x_se_84], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_13.run(buf94, buf96, 960, 8, grid=grid(960), stream=stream0)
        # Topologically Sorted Source Nodes: [x_362, x_se_84, x_se_85], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 16, 1, 1), (16, 1, 1, 1))
        del arg153_1
        del buf96
        buf98 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_362, x_se_84, x_se_85, x_se_86], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_18.run(buf98, arg154_1, 128, grid=grid(128), stream=stream0)
        del arg154_1
        # Topologically Sorted Source Nodes: [x_362, x_se_84, x_se_85, x_se_86, x_se_87], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf99 = extern_kernels.convolution(buf98, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg155_1
        del buf98
        buf100 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_362, x_se_84, x_se_85, x_se_86, x_se_87, hardsigmoid_21, x_363], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_15.run(buf100, buf99, arg156_1, 983040, grid=grid(983040), stream=stream0)
        del arg156_1
        # Topologically Sorted Source Nodes: [x_362, x_se_84, x_se_85, x_se_86, x_se_87, hardsigmoid_21, x_363, x_364], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf101 = extern_kernels.convolution(buf100, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 40, 32, 32), (40960, 1, 1280, 40))
        del arg157_1
        buf102 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_365, x_366], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_20.run(buf102, arg158_1, arg159_1, arg160_1, arg161_1, buf88, 327680, grid=grid(327680), stream=stream0)
        del arg158_1
        del arg159_1
        del arg160_1
        del arg161_1
        del buf88
        # Topologically Sorted Source Nodes: [x_367], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 120, 32, 32), (122880, 1, 3840, 120))
        del arg162_1
        buf104 = buf103; del buf103  # reuse
        buf105 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_368, x_369], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_17.run(buf104, arg163_1, arg164_1, arg165_1, arg166_1, buf105, 983040, grid=grid(983040), stream=stream0)
        del arg163_1
        del arg164_1
        del arg165_1
        del arg166_1
        del buf104
        # Topologically Sorted Source Nodes: [x_369, x_370], Original ATen: [aten.hardswish, aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg167_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf106, (8, 120, 32, 32), (122880, 1, 3840, 120))
        del arg167_1
        del buf105
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_371], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf107, arg168_1, arg169_1, arg170_1, arg171_1, 983040, grid=grid(983040), stream=stream0)
        del arg168_1
        del arg169_1
        del arg170_1
        del arg171_1
        buf108 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_372, x_se_88], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_12.run(buf107, buf108, 7680, 128, grid=grid(7680), stream=stream0)
        buf110 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [x_372, x_se_88], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_13.run(buf108, buf110, 960, 8, grid=grid(960), stream=stream0)
        del buf108
        # Topologically Sorted Source Nodes: [x_372, x_se_88, x_se_89], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf111 = extern_kernels.convolution(buf110, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 16, 1, 1), (16, 1, 1, 1))
        del arg172_1
        del buf110
        buf112 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [x_372, x_se_88, x_se_89, x_se_90], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_18.run(buf112, arg173_1, 128, grid=grid(128), stream=stream0)
        del arg173_1
        # Topologically Sorted Source Nodes: [x_372, x_se_88, x_se_89, x_se_90, x_se_91], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf113 = extern_kernels.convolution(buf112, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg174_1
        del buf112
        buf114 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [x_372, x_se_88, x_se_89, x_se_90, x_se_91, hardsigmoid_22, x_373], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_15.run(buf114, buf113, arg175_1, 983040, grid=grid(983040), stream=stream0)
        del arg175_1
        del buf113
        # Topologically Sorted Source Nodes: [x_372, x_se_88, x_se_89, x_se_90, x_se_91, hardsigmoid_22, x_373, x_374], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf115 = extern_kernels.convolution(buf114, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 40, 32, 32), (40960, 1, 1280, 40))
        del arg176_1
        del buf114
        buf116 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_375, x_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_19.run(buf116, buf115, arg177_1, arg178_1, arg179_1, arg180_1, 327680, grid=grid(327680), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        del buf115
        # Topologically Sorted Source Nodes: [x_375, x_376, x_377], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf117 = extern_kernels.convolution(buf116, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 200, 32, 32), (204800, 1, 6400, 200))
        del arg181_1
        del buf116
        buf118 = buf117; del buf117  # reuse
        buf119 = empty_strided_cuda((8, 200, 32, 32), (204800, 1, 6400, 200), torch.float32)
        # Topologically Sorted Source Nodes: [x_378, x_379], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_21.run(buf118, arg182_1, arg183_1, arg184_1, arg185_1, buf119, 1638400, grid=grid(1638400), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        del buf118
        # Topologically Sorted Source Nodes: [x_379, x_380], Original ATen: [aten.hardswish, aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg186_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=200, bias=None)
        assert_size_stride(buf120, (8, 200, 16, 16), (51200, 1, 3200, 200))
        del arg186_1
        del buf119
        buf121 = buf120; del buf120  # reuse
        buf122 = empty_strided_cuda((8, 200, 16, 16), (51200, 1, 3200, 200), torch.float32)
        # Topologically Sorted Source Nodes: [x_381, x_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf121, arg187_1, arg188_1, arg189_1, arg190_1, buf122, 409600, grid=grid(409600), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        del buf121
        # Topologically Sorted Source Nodes: [x_382, x_383], Original ATen: [aten.hardswish, aten.convolution]
        buf123 = extern_kernels.convolution(buf122, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (8, 72, 16, 16), (18432, 1, 1152, 72))
        del arg191_1
        del buf122
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_384], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_23.run(buf124, arg192_1, arg193_1, arg194_1, arg195_1, 147456, grid=grid(147456), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        # Topologically Sorted Source Nodes: [x_385], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 216, 16, 16), (55296, 1, 3456, 216))
        del arg196_1
        buf126 = buf125; del buf125  # reuse
        buf127 = empty_strided_cuda((8, 216, 16, 16), (55296, 1, 3456, 216), torch.float32)
        # Topologically Sorted Source Nodes: [x_386, x_387], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf126, arg197_1, arg198_1, arg199_1, arg200_1, buf127, 442368, grid=grid(442368), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        del buf126
        # Topologically Sorted Source Nodes: [x_387, x_388], Original ATen: [aten.hardswish, aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg201_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf128, (8, 216, 16, 16), (55296, 1, 3456, 216))
        del arg201_1
        buf129 = buf128; del buf128  # reuse
        buf130 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_389, x_390], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf129, arg202_1, arg203_1, arg204_1, arg205_1, buf130, 442368, grid=grid(442368), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        del buf129
        # Topologically Sorted Source Nodes: [x_390, x_391], Original ATen: [aten.hardswish, aten.convolution]
        buf131 = extern_kernels.convolution(buf130, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 72, 16, 16), (18432, 1, 1152, 72))
        del arg206_1
        buf132 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_392, x_393], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_25.run(buf132, buf131, arg207_1, arg208_1, arg209_1, arg210_1, 147456, grid=grid(147456), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        del buf131
        # Topologically Sorted Source Nodes: [x_394], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 216, 16, 16), (55296, 1, 3456, 216))
        del arg211_1
        buf134 = buf133; del buf133  # reuse
        buf135 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_395, x_396], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf134, arg212_1, arg213_1, arg214_1, arg215_1, buf135, 442368, grid=grid(442368), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        del buf134
        # Topologically Sorted Source Nodes: [x_396, x_397], Original ATen: [aten.hardswish, aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg216_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf136, (8, 216, 16, 16), (55296, 1, 3456, 216))
        del arg216_1
        buf137 = buf136; del buf136  # reuse
        buf138 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [x_398, x_399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf137, arg217_1, arg218_1, arg219_1, arg220_1, buf138, 442368, grid=grid(442368), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        del buf137
        # Topologically Sorted Source Nodes: [x_399, x_400], Original ATen: [aten.hardswish, aten.convolution]
        buf139 = extern_kernels.convolution(buf138, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (8, 72, 16, 16), (18432, 1, 1152, 72))
        del arg221_1
        buf140 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [x_401, x_402], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_25.run(buf140, buf139, arg222_1, arg223_1, arg224_1, arg225_1, 147456, grid=grid(147456), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        del buf139
        # Topologically Sorted Source Nodes: [x_403], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 216, 16, 16), (55296, 1, 3456, 216))
        del arg226_1
        buf142 = buf141; del buf141  # reuse
        buf143 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [x_404, x_405], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf142, arg227_1, arg228_1, arg229_1, arg230_1, buf143, 442368, grid=grid(442368), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        del buf142
        # Topologically Sorted Source Nodes: [x_405, x_406], Original ATen: [aten.hardswish, aten.convolution]
        buf144 = extern_kernels.convolution(buf143, arg231_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf144, (8, 216, 16, 16), (55296, 1, 3456, 216))
        del arg231_1
        buf145 = buf144; del buf144  # reuse
        buf146 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_407, x_408], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf145, arg232_1, arg233_1, arg234_1, arg235_1, buf146, 442368, grid=grid(442368), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        del buf145
        # Topologically Sorted Source Nodes: [x_408, x_409], Original ATen: [aten.hardswish, aten.convolution]
        buf147 = extern_kernels.convolution(buf146, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (8, 72, 16, 16), (18432, 1, 1152, 72))
        del arg236_1
        buf148 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_410, x_411], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_25.run(buf148, buf147, arg237_1, arg238_1, arg239_1, arg240_1, 147456, grid=grid(147456), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        del buf147
        # Topologically Sorted Source Nodes: [x_412], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 216, 16, 16), (55296, 1, 3456, 216))
        del arg241_1
        buf150 = buf149; del buf149  # reuse
        buf151 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_413, x_414], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf150, arg242_1, arg243_1, arg244_1, arg245_1, buf151, 442368, grid=grid(442368), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        del buf150
        # Topologically Sorted Source Nodes: [x_414, x_415], Original ATen: [aten.hardswish, aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg246_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf152, (8, 216, 16, 16), (55296, 1, 3456, 216))
        del arg246_1
        buf153 = buf152; del buf152  # reuse
        buf154 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [x_416, x_417], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf153, arg247_1, arg248_1, arg249_1, arg250_1, buf154, 442368, grid=grid(442368), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        del buf153
        # Topologically Sorted Source Nodes: [x_417, x_418], Original ATen: [aten.hardswish, aten.convolution]
        buf155 = extern_kernels.convolution(buf154, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (8, 72, 16, 16), (18432, 1, 1152, 72))
        del arg251_1
        del buf154
        buf156 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_419, x_420], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_25.run(buf156, buf155, arg252_1, arg253_1, arg254_1, arg255_1, 147456, grid=grid(147456), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        del buf155
        # Topologically Sorted Source Nodes: [x_419, x_420, x_421], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf157 = extern_kernels.convolution(buf156, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 360, 16, 16), (92160, 1, 5760, 360))
        del arg256_1
        del buf156
        buf158 = buf157; del buf157  # reuse
        buf159 = empty_strided_cuda((8, 360, 16, 16), (92160, 1, 5760, 360), torch.float32)
        # Topologically Sorted Source Nodes: [x_422, x_423], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_26.run(buf158, arg257_1, arg258_1, arg259_1, arg260_1, buf159, 737280, grid=grid(737280), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        del buf158
        # Topologically Sorted Source Nodes: [x_423, x_424], Original ATen: [aten.hardswish, aten.convolution]
        buf160 = extern_kernels.convolution(buf159, arg261_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf160, (8, 360, 16, 16), (92160, 1, 5760, 360))
        del arg261_1
        del buf159
        buf161 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_425], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf161, arg262_1, arg263_1, arg264_1, arg265_1, 737280, grid=grid(737280), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        buf162 = empty_strided_cuda((8, 360, 1, 1, 2), (720, 1, 5760, 5760, 360), torch.float32)
        # Topologically Sorted Source Nodes: [x_426, x_se_92], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_28.run(buf161, buf162, 5760, 128, grid=grid(5760), stream=stream0)
        buf164 = empty_strided_cuda((8, 360, 1, 1), (360, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_426, x_se_92], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_29.run(buf162, buf164, 2880, 2, grid=grid(2880), stream=stream0)
        # Topologically Sorted Source Nodes: [x_426, x_se_92, x_se_93], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf165 = extern_kernels.convolution(buf164, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 24, 1, 1), (24, 1, 1, 1))
        del arg266_1
        del buf164
        buf166 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_426, x_se_92, x_se_93, x_se_94], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_30.run(buf166, arg267_1, 192, grid=grid(192), stream=stream0)
        del arg267_1
        # Topologically Sorted Source Nodes: [x_426, x_se_92, x_se_93, x_se_94, x_se_95], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf167 = extern_kernels.convolution(buf166, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 360, 1, 1), (360, 1, 1, 1))
        del arg268_1
        del buf166
        buf168 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_426, x_se_92, x_se_93, x_se_94, x_se_95, hardsigmoid_23, x_427], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31.run(buf168, buf167, arg269_1, 737280, grid=grid(737280), stream=stream0)
        del arg269_1
        # Topologically Sorted Source Nodes: [x_426, x_se_92, x_se_93, x_se_94, x_se_95, hardsigmoid_23, x_427, x_428], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf169 = extern_kernels.convolution(buf168, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 120, 16, 16), (30720, 1, 1920, 120))
        del arg270_1
        buf170 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_429], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf170, arg271_1, arg272_1, arg273_1, arg274_1, 245760, grid=grid(245760), stream=stream0)
        del arg271_1
        del arg272_1
        del arg273_1
        del arg274_1
        # Topologically Sorted Source Nodes: [x_430], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, arg275_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 360, 16, 16), (92160, 1, 5760, 360))
        del arg275_1
        buf172 = buf171; del buf171  # reuse
        buf173 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_431, x_432], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_26.run(buf172, arg276_1, arg277_1, arg278_1, arg279_1, buf173, 737280, grid=grid(737280), stream=stream0)
        del arg276_1
        del arg277_1
        del arg278_1
        del arg279_1
        del buf172
        # Topologically Sorted Source Nodes: [x_432, x_433], Original ATen: [aten.hardswish, aten.convolution]
        buf174 = extern_kernels.convolution(buf173, arg280_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf174, (8, 360, 16, 16), (92160, 1, 5760, 360))
        del arg280_1
        del buf173
        buf175 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [x_434], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf175, arg281_1, arg282_1, arg283_1, arg284_1, 737280, grid=grid(737280), stream=stream0)
        del arg281_1
        del arg282_1
        del arg283_1
        del arg284_1
        buf176 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_435, x_se_96], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_28.run(buf175, buf176, 5760, 128, grid=grid(5760), stream=stream0)
        buf178 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_435, x_se_96], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_29.run(buf176, buf178, 2880, 2, grid=grid(2880), stream=stream0)
        # Topologically Sorted Source Nodes: [x_435, x_se_96, x_se_97], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf179 = extern_kernels.convolution(buf178, arg285_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg285_1
        del buf178
        buf180 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_435, x_se_96, x_se_97, x_se_98], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_33.run(buf180, arg286_1, 256, grid=grid(256), stream=stream0)
        del arg286_1
        # Topologically Sorted Source Nodes: [x_435, x_se_96, x_se_97, x_se_98, x_se_99], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf181 = extern_kernels.convolution(buf180, arg287_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 360, 1, 1), (360, 1, 1, 1))
        del arg287_1
        del buf180
        buf182 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [x_435, x_se_96, x_se_97, x_se_98, x_se_99, hardsigmoid_24, x_436], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31.run(buf182, buf181, arg288_1, 737280, grid=grid(737280), stream=stream0)
        del arg288_1
        # Topologically Sorted Source Nodes: [x_435, x_se_96, x_se_97, x_se_98, x_se_99, hardsigmoid_24, x_436, x_437], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf183 = extern_kernels.convolution(buf182, arg289_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 120, 16, 16), (30720, 1, 1920, 120))
        del arg289_1
        buf184 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [x_438, x_439], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_34.run(buf184, buf183, arg290_1, arg291_1, arg292_1, arg293_1, 245760, grid=grid(245760), stream=stream0)
        del arg290_1
        del arg291_1
        del arg292_1
        del arg293_1
        del buf183
        # Topologically Sorted Source Nodes: [x_440], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 360, 16, 16), (92160, 1, 5760, 360))
        del arg294_1
        buf186 = buf185; del buf185  # reuse
        buf187 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [x_441, x_442], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_26.run(buf186, arg295_1, arg296_1, arg297_1, arg298_1, buf187, 737280, grid=grid(737280), stream=stream0)
        del arg295_1
        del arg296_1
        del arg297_1
        del arg298_1
        del buf186
        # Topologically Sorted Source Nodes: [x_442, x_443], Original ATen: [aten.hardswish, aten.convolution]
        buf188 = extern_kernels.convolution(buf187, arg299_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf188, (8, 360, 16, 16), (92160, 1, 5760, 360))
        del arg299_1
        del buf187
        buf189 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_444], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf189, arg300_1, arg301_1, arg302_1, arg303_1, 737280, grid=grid(737280), stream=stream0)
        del arg300_1
        del arg301_1
        del arg302_1
        del arg303_1
        buf190 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [x_445, x_se_100], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_28.run(buf189, buf190, 5760, 128, grid=grid(5760), stream=stream0)
        buf192 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [x_445, x_se_100], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_29.run(buf190, buf192, 2880, 2, grid=grid(2880), stream=stream0)
        # Topologically Sorted Source Nodes: [x_445, x_se_100, x_se_101], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf193 = extern_kernels.convolution(buf192, arg304_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg304_1
        del buf192
        buf194 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [x_445, x_se_100, x_se_101, x_se_102], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_33.run(buf194, arg305_1, 256, grid=grid(256), stream=stream0)
        del arg305_1
        # Topologically Sorted Source Nodes: [x_445, x_se_100, x_se_101, x_se_102, x_se_103], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf195 = extern_kernels.convolution(buf194, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 360, 1, 1), (360, 1, 1, 1))
        del arg306_1
        del buf194
        buf196 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_445, x_se_100, x_se_101, x_se_102, x_se_103, hardsigmoid_25, x_446], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31.run(buf196, buf195, arg307_1, 737280, grid=grid(737280), stream=stream0)
        del arg307_1
        # Topologically Sorted Source Nodes: [x_445, x_se_100, x_se_101, x_se_102, x_se_103, hardsigmoid_25, x_446, x_447], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf197 = extern_kernels.convolution(buf196, arg308_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 120, 16, 16), (30720, 1, 1920, 120))
        del arg308_1
        buf198 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [x_448, x_449], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_34.run(buf198, buf197, arg309_1, arg310_1, arg311_1, arg312_1, 245760, grid=grid(245760), stream=stream0)
        del arg309_1
        del arg310_1
        del arg311_1
        del arg312_1
        del buf197
        # Topologically Sorted Source Nodes: [x_450], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, arg313_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (8, 360, 16, 16), (92160, 1, 5760, 360))
        del arg313_1
        buf200 = buf199; del buf199  # reuse
        buf201 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [x_451, x_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_26.run(buf200, arg314_1, arg315_1, arg316_1, arg317_1, buf201, 737280, grid=grid(737280), stream=stream0)
        del arg314_1
        del arg315_1
        del arg316_1
        del arg317_1
        del buf200
        # Topologically Sorted Source Nodes: [x_452, x_453], Original ATen: [aten.hardswish, aten.convolution]
        buf202 = extern_kernels.convolution(buf201, arg318_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf202, (8, 360, 16, 16), (92160, 1, 5760, 360))
        del arg318_1
        del buf201
        buf203 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [x_454], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf203, arg319_1, arg320_1, arg321_1, arg322_1, 737280, grid=grid(737280), stream=stream0)
        del arg319_1
        del arg320_1
        del arg321_1
        del arg322_1
        buf204 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_455, x_se_104], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_28.run(buf203, buf204, 5760, 128, grid=grid(5760), stream=stream0)
        buf206 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_455, x_se_104], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_29.run(buf204, buf206, 2880, 2, grid=grid(2880), stream=stream0)
        # Topologically Sorted Source Nodes: [x_455, x_se_104, x_se_105], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf207 = extern_kernels.convolution(buf206, arg323_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg323_1
        del buf206
        buf208 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_455, x_se_104, x_se_105, x_se_106], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_33.run(buf208, arg324_1, 256, grid=grid(256), stream=stream0)
        del arg324_1
        # Topologically Sorted Source Nodes: [x_455, x_se_104, x_se_105, x_se_106, x_se_107], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf209 = extern_kernels.convolution(buf208, arg325_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (8, 360, 1, 1), (360, 1, 1, 1))
        del arg325_1
        del buf208
        buf210 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [x_455, x_se_104, x_se_105, x_se_106, x_se_107, hardsigmoid_26, x_456], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31.run(buf210, buf209, arg326_1, 737280, grid=grid(737280), stream=stream0)
        del arg326_1
        # Topologically Sorted Source Nodes: [x_455, x_se_104, x_se_105, x_se_106, x_se_107, hardsigmoid_26, x_456, x_457], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf211 = extern_kernels.convolution(buf210, arg327_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (8, 120, 16, 16), (30720, 1, 1920, 120))
        del arg327_1
        buf212 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [x_458, x_459], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_34.run(buf212, buf211, arg328_1, arg329_1, arg330_1, arg331_1, 245760, grid=grid(245760), stream=stream0)
        del arg328_1
        del arg329_1
        del arg330_1
        del arg331_1
        del buf211
        # Topologically Sorted Source Nodes: [x_460], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, arg332_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 360, 16, 16), (92160, 1, 5760, 360))
        del arg332_1
        buf214 = buf213; del buf213  # reuse
        buf215 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [x_461, x_462], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_26.run(buf214, arg333_1, arg334_1, arg335_1, arg336_1, buf215, 737280, grid=grid(737280), stream=stream0)
        del arg333_1
        del arg334_1
        del arg335_1
        del arg336_1
        del buf214
        # Topologically Sorted Source Nodes: [x_462, x_463], Original ATen: [aten.hardswish, aten.convolution]
        buf216 = extern_kernels.convolution(buf215, arg337_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf216, (8, 360, 16, 16), (92160, 1, 5760, 360))
        del arg337_1
        del buf215
        buf217 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [x_464], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf217, arg338_1, arg339_1, arg340_1, arg341_1, 737280, grid=grid(737280), stream=stream0)
        del arg338_1
        del arg339_1
        del arg340_1
        del arg341_1
        buf218 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_465, x_se_108], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_28.run(buf217, buf218, 5760, 128, grid=grid(5760), stream=stream0)
        buf220 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [x_465, x_se_108], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_29.run(buf218, buf220, 2880, 2, grid=grid(2880), stream=stream0)
        # Topologically Sorted Source Nodes: [x_465, x_se_108, x_se_109], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf221 = extern_kernels.convolution(buf220, arg342_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg342_1
        del buf220
        buf222 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [x_465, x_se_108, x_se_109, x_se_110], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_33.run(buf222, arg343_1, 256, grid=grid(256), stream=stream0)
        del arg343_1
        # Topologically Sorted Source Nodes: [x_465, x_se_108, x_se_109, x_se_110, x_se_111], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf223 = extern_kernels.convolution(buf222, arg344_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 360, 1, 1), (360, 1, 1, 1))
        del arg344_1
        del buf222
        buf224 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_465, x_se_108, x_se_109, x_se_110, x_se_111, hardsigmoid_27, x_466], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31.run(buf224, buf223, arg345_1, 737280, grid=grid(737280), stream=stream0)
        del arg345_1
        # Topologically Sorted Source Nodes: [x_465, x_se_108, x_se_109, x_se_110, x_se_111, hardsigmoid_27, x_466, x_467], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf225 = extern_kernels.convolution(buf224, arg346_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 120, 16, 16), (30720, 1, 1920, 120))
        del arg346_1
        buf226 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [x_468, x_469], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_34.run(buf226, buf225, arg347_1, arg348_1, arg349_1, arg350_1, 245760, grid=grid(245760), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        del arg350_1
        del buf225
        # Topologically Sorted Source Nodes: [x_470], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, arg351_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 360, 16, 16), (92160, 1, 5760, 360))
        del arg351_1
        buf228 = buf227; del buf227  # reuse
        buf229 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [x_471, x_472], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_26.run(buf228, arg352_1, arg353_1, arg354_1, arg355_1, buf229, 737280, grid=grid(737280), stream=stream0)
        del arg352_1
        del arg353_1
        del arg354_1
        del arg355_1
        del buf228
        # Topologically Sorted Source Nodes: [x_472, x_473], Original ATen: [aten.hardswish, aten.convolution]
        buf230 = extern_kernels.convolution(buf229, arg356_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf230, (8, 360, 16, 16), (92160, 1, 5760, 360))
        del arg356_1
        del buf229
        buf231 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [x_474], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf231, arg357_1, arg358_1, arg359_1, arg360_1, 737280, grid=grid(737280), stream=stream0)
        del arg357_1
        del arg358_1
        del arg359_1
        del arg360_1
        buf232 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [x_475, x_se_112], Original ATen: [aten.hardswish, aten.mean]
        triton_red_fused_hardswish_mean_28.run(buf231, buf232, 5760, 128, grid=grid(5760), stream=stream0)
        buf234 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [x_475, x_se_112], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_29.run(buf232, buf234, 2880, 2, grid=grid(2880), stream=stream0)
        # Topologically Sorted Source Nodes: [x_475, x_se_112, x_se_113], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf235 = extern_kernels.convolution(buf234, arg361_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg361_1
        del buf234
        buf236 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [x_475, x_se_112, x_se_113, x_se_114], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_33.run(buf236, arg362_1, 256, grid=grid(256), stream=stream0)
        del arg362_1
        # Topologically Sorted Source Nodes: [x_475, x_se_112, x_se_113, x_se_114, x_se_115], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf237 = extern_kernels.convolution(buf236, arg363_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 360, 1, 1), (360, 1, 1, 1))
        del arg363_1
        del buf236
        buf238 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [x_475, x_se_112, x_se_113, x_se_114, x_se_115, hardsigmoid_28, x_476], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_31.run(buf238, buf237, arg364_1, 737280, grid=grid(737280), stream=stream0)
        del arg364_1
        del buf237
        # Topologically Sorted Source Nodes: [x_475, x_se_112, x_se_113, x_se_114, x_se_115, hardsigmoid_28, x_476, x_477], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf239 = extern_kernels.convolution(buf238, arg365_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 120, 16, 16), (30720, 1, 1920, 120))
        del arg365_1
        del buf238
        buf240 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [x_478, x_479], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_34.run(buf240, buf239, arg366_1, arg367_1, arg368_1, arg369_1, 245760, grid=grid(245760), stream=stream0)
        del arg366_1
        del arg367_1
        del arg368_1
        del arg369_1
        del buf239
        # Topologically Sorted Source Nodes: [x_478, x_479, x_480], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf241 = extern_kernels.convolution(buf240, arg370_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (8, 720, 16, 16), (184320, 1, 11520, 720))
        del arg370_1
        del buf240
        buf242 = buf241; del buf241  # reuse
        buf243 = empty_strided_cuda((8, 720, 16, 16), (184320, 1, 11520, 720), torch.float32)
        # Topologically Sorted Source Nodes: [x_481, x_482], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_35.run(buf242, arg371_1, arg372_1, arg373_1, arg374_1, buf243, 1474560, grid=grid(1474560), stream=stream0)
        del arg371_1
        del arg372_1
        del arg373_1
        del arg374_1
        del buf242
        # Topologically Sorted Source Nodes: [x_482, x_483], Original ATen: [aten.hardswish, aten.convolution]
        buf244 = extern_kernels.convolution(buf243, arg375_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=720, bias=None)
        assert_size_stride(buf244, (8, 720, 8, 8), (46080, 1, 5760, 720))
        del arg375_1
        del buf243
        buf245 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [x_484], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf245, arg376_1, arg377_1, arg378_1, arg379_1, 368640, grid=grid(368640), stream=stream0)
        del arg376_1
        del arg377_1
        del arg378_1
        del arg379_1
        buf247 = reinterpret_tensor(buf232, (8, 720, 1, 1), (720, 1, 1, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_485, x_se_116], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_37.run(buf245, buf247, 5760, 64, grid=grid(5760), stream=stream0)
        # Topologically Sorted Source Nodes: [x_485, x_se_116, x_se_117], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf248 = extern_kernels.convolution(buf247, arg380_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg380_1
        del buf247
        buf249 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [x_485, x_se_116, x_se_117, x_se_118], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_33.run(buf249, arg381_1, 256, grid=grid(256), stream=stream0)
        del arg381_1
        # Topologically Sorted Source Nodes: [x_485, x_se_116, x_se_117, x_se_118, x_se_119], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf250 = extern_kernels.convolution(buf249, arg382_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (8, 720, 1, 1), (720, 1, 1, 1))
        del arg382_1
        del buf249
        buf251 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [x_485, x_se_116, x_se_117, x_se_118, x_se_119, hardsigmoid_29, x_486], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_38.run(buf251, buf250, arg383_1, 368640, grid=grid(368640), stream=stream0)
        del arg383_1
        del buf250
        # Topologically Sorted Source Nodes: [x_485, x_se_116, x_se_117, x_se_118, x_se_119, hardsigmoid_29, x_486, x_487], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf252 = extern_kernels.convolution(buf251, arg384_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (8, 184, 8, 8), (11776, 1, 1472, 184))
        del arg384_1
        del buf251
        buf253 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [x_488], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_39.run(buf253, arg385_1, arg386_1, arg387_1, arg388_1, 94208, grid=grid(94208), stream=stream0)
        del arg385_1
        del arg386_1
        del arg387_1
        del arg388_1
        # Topologically Sorted Source Nodes: [x_489], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, arg389_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 736, 8, 8), (47104, 1, 5888, 736))
        del arg389_1
        buf255 = buf254; del buf254  # reuse
        buf256 = empty_strided_cuda((8, 736, 8, 8), (47104, 1, 5888, 736), torch.float32)
        # Topologically Sorted Source Nodes: [x_490, x_491], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_40.run(buf255, arg390_1, arg391_1, arg392_1, arg393_1, buf256, 376832, grid=grid(376832), stream=stream0)
        del arg390_1
        del arg391_1
        del arg392_1
        del arg393_1
        del buf255
        # Topologically Sorted Source Nodes: [x_491, x_492], Original ATen: [aten.hardswish, aten.convolution]
        buf257 = extern_kernels.convolution(buf256, arg394_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf257, (8, 736, 8, 8), (47104, 1, 5888, 736))
        del arg394_1
        del buf256
        buf258 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [x_493], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf258, arg395_1, arg396_1, arg397_1, arg398_1, 376832, grid=grid(376832), stream=stream0)
        del arg395_1
        del arg396_1
        del arg397_1
        del arg398_1
        buf260 = empty_strided_cuda((8, 736, 1, 1), (736, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_494, x_se_120], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_42.run(buf258, buf260, 5888, 64, grid=grid(5888), stream=stream0)
        # Topologically Sorted Source Nodes: [x_494, x_se_120, x_se_121], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf261 = extern_kernels.convolution(buf260, arg399_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg399_1
        del buf260
        buf262 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [x_494, x_se_120, x_se_121, x_se_122], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_43.run(buf262, arg400_1, 384, grid=grid(384), stream=stream0)
        del arg400_1
        # Topologically Sorted Source Nodes: [x_494, x_se_120, x_se_121, x_se_122, x_se_123], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf263 = extern_kernels.convolution(buf262, arg401_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (8, 736, 1, 1), (736, 1, 1, 1))
        del arg401_1
        del buf262
        buf264 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [x_494, x_se_120, x_se_121, x_se_122, x_se_123, hardsigmoid_30, x_495], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_44.run(buf264, buf263, arg402_1, 376832, grid=grid(376832), stream=stream0)
        del arg402_1
        # Topologically Sorted Source Nodes: [x_494, x_se_120, x_se_121, x_se_122, x_se_123, hardsigmoid_30, x_495, x_496], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf265 = extern_kernels.convolution(buf264, arg403_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (8, 184, 8, 8), (11776, 1, 1472, 184))
        del arg403_1
        buf266 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [x_497, x_498], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_45.run(buf266, buf265, arg404_1, arg405_1, arg406_1, arg407_1, 94208, grid=grid(94208), stream=stream0)
        del arg404_1
        del arg405_1
        del arg406_1
        del arg407_1
        del buf265
        # Topologically Sorted Source Nodes: [x_499], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, arg408_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (8, 736, 8, 8), (47104, 1, 5888, 736))
        del arg408_1
        buf268 = buf267; del buf267  # reuse
        buf269 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [x_500, x_501], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_40.run(buf268, arg409_1, arg410_1, arg411_1, arg412_1, buf269, 376832, grid=grid(376832), stream=stream0)
        del arg409_1
        del arg410_1
        del arg411_1
        del arg412_1
        del buf268
        # Topologically Sorted Source Nodes: [x_501, x_502], Original ATen: [aten.hardswish, aten.convolution]
        buf270 = extern_kernels.convolution(buf269, arg413_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf270, (8, 736, 8, 8), (47104, 1, 5888, 736))
        del arg413_1
        del buf269
        buf271 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [x_503], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf271, arg414_1, arg415_1, arg416_1, arg417_1, 376832, grid=grid(376832), stream=stream0)
        del arg414_1
        del arg415_1
        del arg416_1
        del arg417_1
        buf273 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [x_504, x_se_124], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_42.run(buf271, buf273, 5888, 64, grid=grid(5888), stream=stream0)
        # Topologically Sorted Source Nodes: [x_504, x_se_124, x_se_125], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf274 = extern_kernels.convolution(buf273, arg418_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg418_1
        del buf273
        buf275 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [x_504, x_se_124, x_se_125, x_se_126], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_43.run(buf275, arg419_1, 384, grid=grid(384), stream=stream0)
        del arg419_1
        # Topologically Sorted Source Nodes: [x_504, x_se_124, x_se_125, x_se_126, x_se_127], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf276 = extern_kernels.convolution(buf275, arg420_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 736, 1, 1), (736, 1, 1, 1))
        del arg420_1
        del buf275
        buf277 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [x_504, x_se_124, x_se_125, x_se_126, x_se_127, hardsigmoid_31, x_505], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_44.run(buf277, buf276, arg421_1, 376832, grid=grid(376832), stream=stream0)
        del arg421_1
        # Topologically Sorted Source Nodes: [x_504, x_se_124, x_se_125, x_se_126, x_se_127, hardsigmoid_31, x_505, x_506], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf278 = extern_kernels.convolution(buf277, arg422_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 184, 8, 8), (11776, 1, 1472, 184))
        del arg422_1
        buf279 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [x_507, x_508], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_45.run(buf279, buf278, arg423_1, arg424_1, arg425_1, arg426_1, 94208, grid=grid(94208), stream=stream0)
        del arg423_1
        del arg424_1
        del arg425_1
        del arg426_1
        del buf278
        # Topologically Sorted Source Nodes: [x_509], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, arg427_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (8, 736, 8, 8), (47104, 1, 5888, 736))
        del arg427_1
        buf281 = buf280; del buf280  # reuse
        buf282 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [x_510, x_511], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_40.run(buf281, arg428_1, arg429_1, arg430_1, arg431_1, buf282, 376832, grid=grid(376832), stream=stream0)
        del arg428_1
        del arg429_1
        del arg430_1
        del arg431_1
        del buf281
        # Topologically Sorted Source Nodes: [x_511, x_512], Original ATen: [aten.hardswish, aten.convolution]
        buf283 = extern_kernels.convolution(buf282, arg432_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf283, (8, 736, 8, 8), (47104, 1, 5888, 736))
        del arg432_1
        del buf282
        buf284 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [x_513], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf284, arg433_1, arg434_1, arg435_1, arg436_1, 376832, grid=grid(376832), stream=stream0)
        del arg433_1
        del arg434_1
        del arg435_1
        del arg436_1
        buf286 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [x_514, x_se_128], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_42.run(buf284, buf286, 5888, 64, grid=grid(5888), stream=stream0)
        # Topologically Sorted Source Nodes: [x_514, x_se_128, x_se_129], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf287 = extern_kernels.convolution(buf286, arg437_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg437_1
        del buf286
        buf288 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [x_514, x_se_128, x_se_129, x_se_130], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_43.run(buf288, arg438_1, 384, grid=grid(384), stream=stream0)
        del arg438_1
        # Topologically Sorted Source Nodes: [x_514, x_se_128, x_se_129, x_se_130, x_se_131], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf289 = extern_kernels.convolution(buf288, arg439_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (8, 736, 1, 1), (736, 1, 1, 1))
        del arg439_1
        del buf288
        buf290 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [x_514, x_se_128, x_se_129, x_se_130, x_se_131, hardsigmoid_32, x_515], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_44.run(buf290, buf289, arg440_1, 376832, grid=grid(376832), stream=stream0)
        del arg440_1
        # Topologically Sorted Source Nodes: [x_514, x_se_128, x_se_129, x_se_130, x_se_131, hardsigmoid_32, x_515, x_516], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf291 = extern_kernels.convolution(buf290, arg441_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 184, 8, 8), (11776, 1, 1472, 184))
        del arg441_1
        buf292 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [x_517, x_518], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_45.run(buf292, buf291, arg442_1, arg443_1, arg444_1, arg445_1, 94208, grid=grid(94208), stream=stream0)
        del arg442_1
        del arg443_1
        del arg444_1
        del arg445_1
        del buf291
        # Topologically Sorted Source Nodes: [x_519], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, arg446_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (8, 736, 8, 8), (47104, 1, 5888, 736))
        del arg446_1
        buf294 = buf293; del buf293  # reuse
        buf295 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [x_520, x_521], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_40.run(buf294, arg447_1, arg448_1, arg449_1, arg450_1, buf295, 376832, grid=grid(376832), stream=stream0)
        del arg447_1
        del arg448_1
        del arg449_1
        del arg450_1
        del buf294
        # Topologically Sorted Source Nodes: [x_521, x_522], Original ATen: [aten.hardswish, aten.convolution]
        buf296 = extern_kernels.convolution(buf295, arg451_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf296, (8, 736, 8, 8), (47104, 1, 5888, 736))
        del arg451_1
        del buf295
        buf297 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [x_523], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf297, arg452_1, arg453_1, arg454_1, arg455_1, 376832, grid=grid(376832), stream=stream0)
        del arg452_1
        del arg453_1
        del arg454_1
        del arg455_1
        buf299 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [x_524, x_se_132], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_42.run(buf297, buf299, 5888, 64, grid=grid(5888), stream=stream0)
        # Topologically Sorted Source Nodes: [x_524, x_se_132, x_se_133], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf300 = extern_kernels.convolution(buf299, arg456_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg456_1
        del buf299
        buf301 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [x_524, x_se_132, x_se_133, x_se_134], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_43.run(buf301, arg457_1, 384, grid=grid(384), stream=stream0)
        del arg457_1
        # Topologically Sorted Source Nodes: [x_524, x_se_132, x_se_133, x_se_134, x_se_135], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf302 = extern_kernels.convolution(buf301, arg458_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (8, 736, 1, 1), (736, 1, 1, 1))
        del arg458_1
        del buf301
        buf303 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [x_524, x_se_132, x_se_133, x_se_134, x_se_135, hardsigmoid_33, x_525], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_44.run(buf303, buf302, arg459_1, 376832, grid=grid(376832), stream=stream0)
        del arg459_1
        # Topologically Sorted Source Nodes: [x_524, x_se_132, x_se_133, x_se_134, x_se_135, hardsigmoid_33, x_525, x_526], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf304 = extern_kernels.convolution(buf303, arg460_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (8, 184, 8, 8), (11776, 1, 1472, 184))
        del arg460_1
        buf305 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [x_527, x_528], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_45.run(buf305, buf304, arg461_1, arg462_1, arg463_1, arg464_1, 94208, grid=grid(94208), stream=stream0)
        del arg461_1
        del arg462_1
        del arg463_1
        del arg464_1
        del buf304
        # Topologically Sorted Source Nodes: [x_529], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, arg465_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (8, 736, 8, 8), (47104, 1, 5888, 736))
        del arg465_1
        buf307 = buf306; del buf306  # reuse
        buf308 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [x_530, x_531], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_40.run(buf307, arg466_1, arg467_1, arg468_1, arg469_1, buf308, 376832, grid=grid(376832), stream=stream0)
        del arg466_1
        del arg467_1
        del arg468_1
        del arg469_1
        del buf307
        # Topologically Sorted Source Nodes: [x_531, x_532], Original ATen: [aten.hardswish, aten.convolution]
        buf309 = extern_kernels.convolution(buf308, arg470_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf309, (8, 736, 8, 8), (47104, 1, 5888, 736))
        del arg470_1
        del buf308
        buf310 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [x_533], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf310, arg471_1, arg472_1, arg473_1, arg474_1, 376832, grid=grid(376832), stream=stream0)
        del arg471_1
        del arg472_1
        del arg473_1
        del arg474_1
        buf312 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [x_534, x_se_136], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_42.run(buf310, buf312, 5888, 64, grid=grid(5888), stream=stream0)
        # Topologically Sorted Source Nodes: [x_534, x_se_136, x_se_137], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf313 = extern_kernels.convolution(buf312, arg475_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg475_1
        del buf312
        buf314 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [x_534, x_se_136, x_se_137, x_se_138], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_43.run(buf314, arg476_1, 384, grid=grid(384), stream=stream0)
        del arg476_1
        # Topologically Sorted Source Nodes: [x_534, x_se_136, x_se_137, x_se_138, x_se_139], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf315 = extern_kernels.convolution(buf314, arg477_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (8, 736, 1, 1), (736, 1, 1, 1))
        del arg477_1
        del buf314
        buf316 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [x_534, x_se_136, x_se_137, x_se_138, x_se_139, hardsigmoid_34, x_535], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_44.run(buf316, buf315, arg478_1, 376832, grid=grid(376832), stream=stream0)
        del arg478_1
        del buf315
        # Topologically Sorted Source Nodes: [x_534, x_se_136, x_se_137, x_se_138, x_se_139, hardsigmoid_34, x_535, x_536], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf317 = extern_kernels.convolution(buf316, arg479_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (8, 184, 8, 8), (11776, 1, 1472, 184))
        del arg479_1
        del buf316
        buf318 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [x_537, x_538], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_45.run(buf318, buf317, arg480_1, arg481_1, arg482_1, arg483_1, 94208, grid=grid(94208), stream=stream0)
        del arg480_1
        del arg481_1
        del arg482_1
        del arg483_1
        del buf317
        # Topologically Sorted Source Nodes: [x_537, x_538, x_539], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf319 = extern_kernels.convolution(buf318, arg484_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (8, 1104, 8, 8), (70656, 1, 8832, 1104))
        del arg484_1
        del buf318
        buf320 = buf319; del buf319  # reuse
        buf321 = empty_strided_cuda((8, 1104, 8, 8), (70656, 1, 8832, 1104), torch.float32)
        # Topologically Sorted Source Nodes: [x_540, x_541], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_46.run(buf320, arg485_1, arg486_1, arg487_1, arg488_1, buf321, 565248, grid=grid(565248), stream=stream0)
        del arg485_1
        del arg486_1
        del arg487_1
        del arg488_1
        del buf320
        # Topologically Sorted Source Nodes: [x_541, x_542], Original ATen: [aten.hardswish, aten.convolution]
        buf322 = extern_kernels.convolution(buf321, arg489_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf322, (8, 1104, 8, 8), (70656, 1, 8832, 1104))
        del arg489_1
        del buf321
        buf323 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [x_543], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_47.run(buf323, arg490_1, arg491_1, arg492_1, arg493_1, 565248, grid=grid(565248), stream=stream0)
        del arg490_1
        del arg491_1
        del arg492_1
        del arg493_1
        buf325 = empty_strided_cuda((8, 1104, 1, 1), (1104, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_544, x_se_140], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_48.run(buf323, buf325, 8832, 64, grid=grid(8832), stream=stream0)
        # Topologically Sorted Source Nodes: [x_544, x_se_140, x_se_141], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf326 = extern_kernels.convolution(buf325, arg494_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg494_1
        del buf325
        buf327 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [x_544, x_se_140, x_se_141, x_se_142], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        triton_poi_fused_convolution_hardswish_mean_43.run(buf327, arg495_1, 384, grid=grid(384), stream=stream0)
        del arg495_1
        # Topologically Sorted Source Nodes: [x_544, x_se_140, x_se_141, x_se_142, x_se_143], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf328 = extern_kernels.convolution(buf327, arg496_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (8, 1104, 1, 1), (1104, 1, 1, 1))
        del arg496_1
        del buf327
        buf329 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [x_544, x_se_140, x_se_141, x_se_142, x_se_143, hardsigmoid_35, x_545], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_49.run(buf329, buf328, arg497_1, 565248, grid=grid(565248), stream=stream0)
        del arg497_1
        del buf328
        # Topologically Sorted Source Nodes: [x_544, x_se_140, x_se_141, x_se_142, x_se_143, hardsigmoid_35, x_545, x_546], Original ATen: [aten.hardswish, aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        buf330 = extern_kernels.convolution(buf329, arg498_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (8, 224, 8, 8), (14336, 1, 1792, 224))
        del arg498_1
        del buf329
        buf331 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [x_547], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_50.run(buf331, arg499_1, arg500_1, arg501_1, arg502_1, 114688, grid=grid(114688), stream=stream0)
        del arg499_1
        del arg500_1
        del arg501_1
        del arg502_1
        # Topologically Sorted Source Nodes: [x_547, x_548], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf332 = extern_kernels.convolution(buf331, arg503_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (8, 1344, 8, 8), (86016, 1, 10752, 1344))
        del arg503_1
        del buf331
        buf333 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [x_549], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_51.run(buf333, arg504_1, arg505_1, arg506_1, arg507_1, 688128, grid=grid(688128), stream=stream0)
        del arg504_1
        del arg505_1
        del arg506_1
        del arg507_1
        buf335 = empty_strided_cuda((8, 1344, 1, 1), (1344, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_550, x_551], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_52.run(buf333, buf335, 10752, 64, grid=grid(10752), stream=stream0)
        del buf333
        # Topologically Sorted Source Nodes: [x_550, x_551, x_552], Original ATen: [aten.hardswish, aten.mean, aten.convolution]
        buf336 = extern_kernels.convolution(buf335, arg508_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (8, 1984, 1, 1), (1984, 1, 1, 1))
        del arg508_1
        del buf335
        buf337 = reinterpret_tensor(buf336, (8, 1984, 1, 1), (1984, 1, 15872, 15872), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [x_553], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_53.run(buf337, 15872, grid=grid(15872), stream=stream0)
        buf338 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_555], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg510_1, reinterpret_tensor(buf337, (8, 1984), (1984, 1), 0), reinterpret_tensor(arg509_1, (1984, 1000), (1, 1984), 0), alpha=1, beta=1, out=buf338)
        del arg509_1
        del arg510_1
        del buf337
    return (buf338, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
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
    arg16_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((120, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((8, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((120, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((200, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((200, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((72, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((360, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((360, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((24, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((360, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((720, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((720, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((32, 720, 1, 1), (720, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((720, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((184, 720, 1, 1), (720, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((48, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((1104, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((224, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((1984, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((1000, 1984), (1984, 1), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('fbnetv3_b', benchmark_compiled_module)
