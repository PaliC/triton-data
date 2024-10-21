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
# Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_152 => convolution_50
# Graph fragment:
#   %convolution_50 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/f2/cf2edtjsbopk3t6evc35g2xgtzmtmejwux6pvmrlfmjs3eqd2dlb.py
# Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_152 => convolution_50
# Graph fragment:
#   %convolution_50 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 72
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


# kernel path: /tmp/torchinductor_sahanp/s6/cs6zbbo425osbh6yshdus6ed6bmsyrprwzioxjb6hkg4mblea4sm.py
# Topologically Sorted Source Nodes: [x_153, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_153 => add_95, mul_159, mul_160, sub_42
#   x_154 => mul_161, sigmoid_40
# Graph fragment:
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_50, %unsqueeze_305), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_307), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_159, %unsqueeze_309), kwargs = {})
#   %add_95 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_160, %unsqueeze_311), kwargs = {})
#   %sigmoid_40 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_95,), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_95, %sigmoid_40), kwargs = {})
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
    xnumel = 3145728
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xr/cxrqqehxapzi57svfqrvs4ogguqmwcrxriam44ujucxfkvvojq3z.py
# Topologically Sorted Source Nodes: [x_154, x_155], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   x_154 => mul_161, sigmoid_40
#   x_155 => convolution_51
# Graph fragment:
#   %sigmoid_40 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_95,), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_95, %sigmoid_40), kwargs = {})
#   %convolution_51 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_161, %arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_3 = async_compile.triton('triton_poi_fused_convolution_silu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (216*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ww/cwwxfz6wkntybcoyedvsxwjvgmiqlgq3hgxzj4naw7kwak5vewhg.py
# Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_156 => add_97, mul_163, mul_164, sub_43
#   x_157 => mul_165, sigmoid_41
# Graph fragment:
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_51, %unsqueeze_313), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_315), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %unsqueeze_317), kwargs = {})
#   %add_97 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %unsqueeze_319), kwargs = {})
#   %sigmoid_41 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_97,), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_97, %sigmoid_41), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /tmp/torchinductor_sahanp/6c/c6cgabnzualumgn6n7f3wh2wenjpghoqtmuxgoxvyydnqakmmiwi.py
# Topologically Sorted Source Nodes: [x_157, x_158], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   x_157 => mul_165, sigmoid_41
#   x_158 => convolution_52
# Graph fragment:
#   %sigmoid_41 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_97,), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_97, %sigmoid_41), kwargs = {})
#   %convolution_52 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_165, %arg11_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_5 = async_compile.triton('triton_poi_fused_convolution_silu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
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


# kernel path: /tmp/torchinductor_sahanp/ds/cds5otk2sqjgy7mok4ufayypy25ynvooctidffzjke4kaqndlvps.py
# Topologically Sorted Source Nodes: [x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_159 => add_99, mul_167, mul_168, sub_44
#   x_160 => mul_169, sigmoid_42
# Graph fragment:
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_321), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %unsqueeze_323), kwargs = {})
#   %mul_168 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_167, %unsqueeze_325), kwargs = {})
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_168, %unsqueeze_327), kwargs = {})
#   %sigmoid_42 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_99,), kwargs = {})
#   %mul_169 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_99, %sigmoid_42), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fq/cfqa5z5nomhiaqerjujnnv4hutolxmqhft5rrprq5jj73ebbu3bg.py
# Topologically Sorted Source Nodes: [x_163, x_164], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   x_163 => mul_173, sigmoid_43
#   x_164 => convolution_54
# Graph fragment:
#   %sigmoid_43 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_101,), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_101, %sigmoid_43), kwargs = {})
#   %convolution_54 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_173, %arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_7 = async_compile.triton('triton_poi_fused_convolution_silu_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
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


# kernel path: /tmp/torchinductor_sahanp/2y/c2yvank77uitgxvadeniw7uahm6xykrfwig4w6dnjkwsa3wncnvo.py
# Topologically Sorted Source Nodes: [x_165], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_165 => add_103, mul_175, mul_176, sub_46
# Graph fragment:
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_54, %unsqueeze_337), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_339), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %unsqueeze_341), kwargs = {})
#   %add_103 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %unsqueeze_343), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lx/clxzrbkxqg4tth6yfvrh67u5xyw726xcgqkm3xs54nsmvze23w62.py
# Topologically Sorted Source Nodes: [x_166, x_se_24], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_166 => mul_177, sigmoid_44
#   x_se_24 => mean_7
# Graph fragment:
#   %sigmoid_44 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_103,), kwargs = {})
#   %mul_177 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_103, %sigmoid_44), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_177, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_9 = async_compile.triton('triton_red_fused_mean_silu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_9(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5p/c5py24lnmhx533wasdfpn5tdzrws45phmmtstfjngijbeot5zmxa.py
# Topologically Sorted Source Nodes: [x_166, x_se_24], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_166 => mul_177, sigmoid_44
#   x_se_24 => mean_7
# Graph fragment:
#   %sigmoid_44 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_103,), kwargs = {})
#   %mul_177 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_103, %sigmoid_44), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_177, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_10 = async_compile.triton('triton_per_fused_mean_silu_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_10(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (2048*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 4096.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/va/cvarxh2m2uurvkjt3mwyrdrlal2chxdcssxc2uu43j5ctagupldl.py
# Topologically Sorted Source Nodes: [x_166, x_se_24, x_se_25, x_se_26], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_166 => mul_177, sigmoid_44
#   x_se_24 => mean_7
#   x_se_25 => convolution_55
#   x_se_26 => relu_6
# Graph fragment:
#   %sigmoid_44 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_103,), kwargs = {})
#   %mul_177 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_103, %sigmoid_44), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_177, [2, 3], True), kwargs = {})
#   %convolution_55 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_7, %arg26_1, %arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_55,), kwargs = {})
triton_poi_fused_convolution_mean_relu_silu_11 = async_compile.triton('triton_poi_fused_convolution_mean_relu_silu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_silu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_silu_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eh/cehdwxpcvp5ffr4523tgytlkoyjq4ilbykfjk4gsuqilnstsub2k.py
# Topologically Sorted Source Nodes: [x_166, x_se_24, x_se_25, x_se_26, x_se_27, sigmoid_6, x_167], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_6 => sigmoid_45
#   x_166 => mul_177, sigmoid_44
#   x_167 => mul_178
#   x_se_24 => mean_7
#   x_se_25 => convolution_55
#   x_se_26 => relu_6
#   x_se_27 => convolution_56
# Graph fragment:
#   %sigmoid_44 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_103,), kwargs = {})
#   %mul_177 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_103, %sigmoid_44), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_177, [2, 3], True), kwargs = {})
#   %convolution_55 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_7, %arg26_1, %arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_55,), kwargs = {})
#   %convolution_56 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %arg28_1, %arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_45 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_56,), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_177, %sigmoid_45), kwargs = {})
triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_12 = async_compile.triton('triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_12(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 64
    x2 = (xindex // 262144)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/am/camu7oyoeyo7wkca4js22wxwew6hsoy6ere3kpji4jdrfgmfnnjs.py
# Topologically Sorted Source Nodes: [x_169, x_171, x_172, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# Source node to ATen node mapping:
#   input_11 => mul_185, sigmoid_46
#   x_169 => add_105, mul_180, mul_181, sub_47
#   x_171 => add_107, mul_183, mul_184, sub_48
#   x_172 => add_108
# Graph fragment:
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_57, %unsqueeze_345), kwargs = {})
#   %mul_180 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %unsqueeze_347), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_180, %unsqueeze_349), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_181, %unsqueeze_351), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_58, %unsqueeze_353), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_355), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_183, %unsqueeze_357), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_184, %unsqueeze_359), kwargs = {})
#   %add_108 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_105, %add_107), kwargs = {})
#   %sigmoid_46 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_108,), kwargs = {})
#   %mul_185 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_108, %sigmoid_46), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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
    tmp30 = tl.sigmoid(tmp29)
    tmp31 = tmp29 * tmp30
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jr/cjrzdwewadi4yfdh6kmoclqqwtsjdvrzouqshwdgx5lwfvxlou3f.py
# Topologically Sorted Source Nodes: [x_181, x_182, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# Source node to ATen node mapping:
#   input_12 => mul_198, sigmoid_50
#   x_181 => add_114, mul_196, mul_197, sub_51
#   x_182 => add_115
# Graph fragment:
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_63, %unsqueeze_377), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_379), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_196, %unsqueeze_381), kwargs = {})
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_197, %unsqueeze_383), kwargs = {})
#   %add_115 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_114, %mul_185), kwargs = {})
#   %sigmoid_50 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_115,), kwargs = {})
#   %mul_198 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_115, %sigmoid_50), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
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
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bg/cbghlr2eercfiw6uq24mayosildlsjarrwt4u7xazjqjuqlllesv.py
# Topologically Sorted Source Nodes: [x_184, x_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_184 => add_117, mul_200, mul_201, sub_52
#   x_185 => mul_202, sigmoid_51
# Graph fragment:
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_64, %unsqueeze_385), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_387), kwargs = {})
#   %mul_201 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_200, %unsqueeze_389), kwargs = {})
#   %add_117 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_201, %unsqueeze_391), kwargs = {})
#   %sigmoid_51 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_117,), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_117, %sigmoid_51), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lb/clbkfxtorehomlik67a5nu6br7neldiujsnscxajeeijiyhjnbdm.py
# Topologically Sorted Source Nodes: [x_185, x_186], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   x_185 => mul_202, sigmoid_51
#   x_186 => convolution_65
# Graph fragment:
#   %sigmoid_51 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_117,), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_117, %sigmoid_51), kwargs = {})
#   %convolution_65 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_202, %arg64_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_16 = async_compile.triton('triton_poi_fused_convolution_silu_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
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


# kernel path: /tmp/torchinductor_sahanp/w5/cw56a5axmi3kh66onip5zv76ohayntdc7ofvq7g3vegi37lndiv3.py
# Topologically Sorted Source Nodes: [x_187], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_187 => add_119, mul_204, mul_205, sub_53
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_65, %unsqueeze_393), kwargs = {})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_395), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_204, %unsqueeze_397), kwargs = {})
#   %add_119 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_205, %unsqueeze_399), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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


# kernel path: /tmp/torchinductor_sahanp/bt/cbtfqfyoukxpzron74htvoydwoeoiwjmooijzol7nshpov4hvv5m.py
# Topologically Sorted Source Nodes: [x_188, x_se_32], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_188 => mul_206, sigmoid_52
#   x_se_32 => mean_9
# Graph fragment:
#   %sigmoid_52 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_119,), kwargs = {})
#   %mul_206 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_119, %sigmoid_52), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_206, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_18 = async_compile.triton('triton_red_fused_mean_silu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_18(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pg/cpgku5fmwle52zstfe6ep4xcmbgrbwsshe5xnr5go3xrhshdfbyx.py
# Topologically Sorted Source Nodes: [x_188, x_se_32], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_188 => mul_206, sigmoid_52
#   x_se_32 => mean_9
# Graph fragment:
#   %sigmoid_52 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_119,), kwargs = {})
#   %mul_206 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_119, %sigmoid_52), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_206, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_19 = async_compile.triton('triton_per_fused_mean_silu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_19(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (1024*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 1024.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/r7/cr73wgfojnm633mcawpk3wdbsgl6msbiny6tb5zdbxq5i3o6vxte.py
# Topologically Sorted Source Nodes: [x_188, x_se_32, x_se_33, x_se_34, x_se_35, sigmoid_8, x_189], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_8 => sigmoid_53
#   x_188 => mul_206, sigmoid_52
#   x_189 => mul_207
#   x_se_32 => mean_9
#   x_se_33 => convolution_66
#   x_se_34 => relu_8
#   x_se_35 => convolution_67
# Graph fragment:
#   %sigmoid_52 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_119,), kwargs = {})
#   %mul_206 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_119, %sigmoid_52), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_206, [2, 3], True), kwargs = {})
#   %convolution_66 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_9, %arg69_1, %arg70_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_66,), kwargs = {})
#   %convolution_67 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg71_1, %arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_53 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_67,), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_206, %sigmoid_53), kwargs = {})
triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_20 = async_compile.triton('triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_20(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 128
    x2 = (xindex // 131072)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (128*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/i7/ci73js7reeudtwhit4gzyxb6t57mbbkbkmx3cu2dmspju3cnpmj7.py
# Topologically Sorted Source Nodes: [x_191, x_193, x_194, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# Source node to ATen node mapping:
#   input_13 => mul_214, sigmoid_54
#   x_191 => add_121, mul_209, mul_210, sub_54
#   x_193 => add_123, mul_212, mul_213, sub_55
#   x_194 => add_124
# Graph fragment:
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_401), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_403), kwargs = {})
#   %mul_210 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_209, %unsqueeze_405), kwargs = {})
#   %add_121 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_210, %unsqueeze_407), kwargs = {})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_69, %unsqueeze_409), kwargs = {})
#   %mul_212 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_411), kwargs = {})
#   %mul_213 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_212, %unsqueeze_413), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_213, %unsqueeze_415), kwargs = {})
#   %add_124 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_121, %add_123), kwargs = {})
#   %sigmoid_54 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_124,), kwargs = {})
#   %mul_214 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_124, %sigmoid_54), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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
    tmp30 = tl.sigmoid(tmp29)
    tmp31 = tmp29 * tmp30
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6l/c6lp7xpelsxwlziief24k6bv7wkr4jn47ssc5oxtrzlho3l2kvya.py
# Topologically Sorted Source Nodes: [x_196, x_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_196 => add_126, mul_216, mul_217, sub_56
#   x_197 => mul_218, sigmoid_55
# Graph fragment:
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_70, %unsqueeze_417), kwargs = {})
#   %mul_216 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %unsqueeze_419), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_216, %unsqueeze_421), kwargs = {})
#   %add_126 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_217, %unsqueeze_423), kwargs = {})
#   %sigmoid_55 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_126,), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_126, %sigmoid_55), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/54/c54gqf4kwyh4cyn27lnel3jbkyb4pkwmjigbzhvvmzuluxpewpj6.py
# Topologically Sorted Source Nodes: [x_203, x_204, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# Source node to ATen node mapping:
#   input_14 => mul_227, sigmoid_58
#   x_203 => add_130, mul_225, mul_226, sub_58
#   x_204 => add_131
# Graph fragment:
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_74, %unsqueeze_433), kwargs = {})
#   %mul_225 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_435), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_225, %unsqueeze_437), kwargs = {})
#   %add_130 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_226, %unsqueeze_439), kwargs = {})
#   %add_131 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_130, %mul_214), kwargs = {})
#   %sigmoid_58 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_131,), kwargs = {})
#   %mul_227 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_131, %sigmoid_58), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lq/clqs3q5ktpezeweogjmb52g4jptyspmt3eznxjjzhfraviygbotw.py
# Topologically Sorted Source Nodes: [reshape_48], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   reshape_48 => clone_29
# Graph fragment:
#   %clone_29 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_12,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_24 = async_compile.triton('triton_poi_fused_clone_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (393216*y1)), xmask)
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/iv/civqvsnm7ivdq7akjn6st2tskdjq67w73lj3cpmo2lepwlxrjho4.py
# Topologically Sorted Source Nodes: [k_9], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   k_9 => clone_30
# Graph fragment:
#   %clone_30 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_13,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_25 = async_compile.triton('triton_poi_fused_clone_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_25(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (128 + y0 + (384*x2) + (393216*y1)), xmask)
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/k6/ck6sx35ruum4g44cp3wwrau5zoli4y44zjvlvdizbej6dbwt7fkd.py
# Topologically Sorted Source Nodes: [x_213], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_213 => clone_33
# Graph fragment:
#   %clone_33 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_37,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_26 = async_compile.triton('triton_poi_fused_clone_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 32
    x3 = (xindex // 32)
    y0 = yindex % 32
    y1 = (yindex // 32)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x3) + (1024*x2) + (1024*((y0 + (32*x3)) // 1024)) + (32768*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (1024*y4)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qp/cqpvu4s4sylnrqojwvp6uwf2o6qzsqvvelnyvvntacdb4ophb4wo.py
# Topologically Sorted Source Nodes: [x_209], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_209 => clone_32
# Graph fragment:
#   %clone_32 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_103,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_27 = async_compile.triton('triton_poi_fused_clone_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (32768*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wo/cwoicvfewgp3zkjjna36zhm4slobfbw4ve73hoqchricjm4finhw.py
# Topologically Sorted Source Nodes: [mul_14, attn_8, attn_9], Original ATen: [aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_8 => add_135
#   attn_9 => amax_4, div_4, exp_4, sub_60, sum_5
#   mul_14 => mul_232
# Graph fragment:
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_8, 0.1767766952966369), kwargs = {})
#   %add_135 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_232, %view_116), kwargs = {})
#   %amax_4 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_135, [-1], True), kwargs = {})
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_135, %amax_4), kwargs = {})
#   %exp_4 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_60,), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_4, [-1], True), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_4, %sum_5), kwargs = {})
triton_red_fused__softmax_add_mul_28 = async_compile.triton('triton_red_fused__softmax_add_mul_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_add_mul_28(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp24 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.1767766952966369
        tmp2 = tmp0 * tmp1
        tmp3 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp4 = tl.full([1, 1], 2048, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp7 = tl.full([1, 1], 63, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((63*(triton_helpers.div_floor_integer(31 + (63*(x0 // 32)) + (r2 // 32),  64))) + (2016*(x0 % 32)) + (64512*x1) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp5, tmp10, tmp11)
        tmp13 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp14 = tmp13 < tmp4
        tmp15 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp16 = tmp15 < tmp7
        tmp17 = tmp16 & tmp14
        tmp18 = tl.load(in_ptr2 + ((63*((31 + (63*(x0 % 32)) + (r2 % 32)) // 64)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp17, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
        tmp20 = tl.where(tmp14, tmp18, tmp19)
        tmp21 = tmp12 + tmp20
        tmp22 = tmp2 + tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = triton_helpers.maximum(_tmp24, tmp23)
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
    tmp24 = triton_helpers.max2(_tmp24, 1)[:, None]
    _tmp52 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp26 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = 0.1767766952966369
        tmp28 = tmp26 * tmp27
        tmp29 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp30 = tl.full([1, 1], 2048, tl.int64)
        tmp31 = tmp29 < tmp30
        tmp32 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp33 = tl.full([1, 1], 63, tl.int64)
        tmp34 = tmp32 < tmp33
        tmp35 = tmp34 & tmp31
        tmp36 = tl.load(in_ptr1 + ((63*(triton_helpers.div_floor_integer(31 + (63*(x0 // 32)) + (r2 // 32),  64))) + (2016*(x0 % 32)) + (64512*x1) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp35, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
        tmp38 = tl.where(tmp31, tmp36, tmp37)
        tmp39 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp40 = tmp39 < tmp30
        tmp41 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp42 = tmp41 < tmp33
        tmp43 = tmp42 & tmp40
        tmp44 = tl.load(in_ptr2 + ((63*((31 + (63*(x0 % 32)) + (r2 % 32)) // 64)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp43, eviction_policy='evict_last', other=0.0)
        tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
        tmp46 = tl.where(tmp40, tmp44, tmp45)
        tmp47 = tmp38 + tmp46
        tmp48 = tmp28 + tmp47
        tmp49 = tmp48 - tmp24
        tmp50 = tl_math.exp(tmp49)
        tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
        tmp53 = _tmp52 + tmp51
        _tmp52 = tl.where(rmask, tmp53, _tmp52)
    tmp52 = tl.sum(_tmp52, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp54 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp55 = 0.1767766952966369
        tmp56 = tmp54 * tmp55
        tmp57 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp58 = tl.full([1, 1], 2048, tl.int64)
        tmp59 = tmp57 < tmp58
        tmp60 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp61 = tl.full([1, 1], 63, tl.int64)
        tmp62 = tmp60 < tmp61
        tmp63 = tmp62 & tmp59
        tmp64 = tl.load(in_ptr1 + ((63*(triton_helpers.div_floor_integer(31 + (63*(x0 // 32)) + (r2 // 32),  64))) + (2016*(x0 % 32)) + (64512*x1) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp63, eviction_policy='evict_last', other=0.0)
        tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
        tmp66 = tl.where(tmp59, tmp64, tmp65)
        tmp67 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp68 = tmp67 < tmp58
        tmp69 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp70 = tmp69 < tmp61
        tmp71 = tmp70 & tmp68
        tmp72 = tl.load(in_ptr2 + ((63*((31 + (63*(x0 % 32)) + (r2 % 32)) // 64)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp71, eviction_policy='evict_last', other=0.0)
        tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
        tmp74 = tl.where(tmp68, tmp72, tmp73)
        tmp75 = tmp66 + tmp74
        tmp76 = tmp56 + tmp75
        tmp77 = tmp76 - tmp24
        tmp78 = tl_math.exp(tmp77)
        tmp79 = tmp78 / tmp52
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp79, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pm/cpmicmcw37z6vur55pi3fnu54opdusbgxlfmsqwe7na7ikdtwkqt.py
# Topologically Sorted Source Nodes: [reshape_50], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   reshape_50 => clone_31
# Graph fragment:
#   %clone_31 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_14,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_29 = async_compile.triton('triton_poi_fused_clone_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (256 + y0 + (384*x2) + (393216*y1)), xmask)
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/su/csutobl4wtwecy6i5pbtw72zoxe3irlxnlwa3gbrp5hzbu5byayr.py
# Topologically Sorted Source Nodes: [x_217, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_217 => add_137, mul_234, mul_235, sub_61
#   x_218 => mul_236, sigmoid_60
# Graph fragment:
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_120, %unsqueeze_449), kwargs = {})
#   %mul_234 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_451), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_234, %unsqueeze_453), kwargs = {})
#   %add_137 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_235, %unsqueeze_455), kwargs = {})
#   %sigmoid_60 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_137,), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, %sigmoid_60), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128) % 1024
    x2 = (xindex // 131072)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (32768*((x1 + (1024*x0)) // 32768)) + (131072*x2) + (x0 % 32)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr1 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/q5/cq5zqlyfqzpcpqssmyfx6oobio6cun44uzfyhnzkybk2p2kvuwvu.py
# Topologically Sorted Source Nodes: [x_223, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_223 => add_142, mul_242, mul_243, sub_63
#   x_224 => mul_244, sigmoid_62
# Graph fragment:
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_78, %unsqueeze_465), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_467), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_242, %unsqueeze_469), kwargs = {})
#   %add_142 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_243, %unsqueeze_471), kwargs = {})
#   %sigmoid_62 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_142,), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_142, %sigmoid_62), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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


# kernel path: /tmp/torchinductor_sahanp/wn/cwnppszc7ai6c4ggw5usnkumg6bof4mtde3f2ie6556x3pwb5ln7.py
# Topologically Sorted Source Nodes: [x_224, x_225], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   x_224 => mul_244, sigmoid_62
#   x_225 => convolution_79
# Graph fragment:
#   %sigmoid_62 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_142,), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_142, %sigmoid_62), kwargs = {})
#   %convolution_79 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_244, %arg124_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_silu_32 = async_compile.triton('triton_poi_fused_convolution_silu_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ph/cphrxt5hesd3wwbrxvdrrxhqukm4g4guxyqgacvatn7li6kfawvj.py
# Topologically Sorted Source Nodes: [x_226], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_226 => add_144, mul_246, mul_247, sub_64
# Graph fragment:
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_79, %unsqueeze_473), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_475), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_246, %unsqueeze_477), kwargs = {})
#   %add_144 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_247, %unsqueeze_479), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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


# kernel path: /tmp/torchinductor_sahanp/gd/cgdgmiovwqp4vtuz5fccuwqd4vosx6s3pxr4iq3uv22bmbe25db3.py
# Topologically Sorted Source Nodes: [x_227, x_se_40], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_227 => mul_248, sigmoid_63
#   x_se_40 => mean_11
# Graph fragment:
#   %sigmoid_63 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_144,), kwargs = {})
#   %mul_248 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_144, %sigmoid_63), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_248, [2, 3], True), kwargs = {})
triton_red_fused_mean_silu_34 = async_compile.triton('triton_red_fused_mean_silu_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_silu_34(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sj/csjt6dm2svmwh4n76h7zjyiic4fcs5uzq5eg62asxholjce3g5eq.py
# Topologically Sorted Source Nodes: [x_227, x_se_40], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_227 => mul_248, sigmoid_63
#   x_se_40 => mean_11
# Graph fragment:
#   %sigmoid_63 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_144,), kwargs = {})
#   %mul_248 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_144, %sigmoid_63), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_248, [2, 3], True), kwargs = {})
triton_per_fused_mean_silu_35 = async_compile.triton('triton_per_fused_mean_silu_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_35(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (512*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 256.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/do/cdok2ztlwnmyrihtfee52uagyhfhxdbhhikoc4xhbsoul52usvvi.py
# Topologically Sorted Source Nodes: [x_227, x_se_40, x_se_41, x_se_42], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x_227 => mul_248, sigmoid_63
#   x_se_40 => mean_11
#   x_se_41 => convolution_80
#   x_se_42 => relu_10
# Graph fragment:
#   %sigmoid_63 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_144,), kwargs = {})
#   %mul_248 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_144, %sigmoid_63), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_248, [2, 3], True), kwargs = {})
#   %convolution_80 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_11, %arg129_1, %arg130_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_80,), kwargs = {})
triton_poi_fused_convolution_mean_relu_silu_36 = async_compile.triton('triton_poi_fused_convolution_mean_relu_silu_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_silu_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_relu_silu_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cb/ccbeknyncb3iv427dvupnvd3ba6lbpq6wc4rsd6rfro7ncjvmik6.py
# Topologically Sorted Source Nodes: [x_227, x_se_40, x_se_41, x_se_42, x_se_43, sigmoid_10, x_228], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   sigmoid_10 => sigmoid_64
#   x_227 => mul_248, sigmoid_63
#   x_228 => mul_249
#   x_se_40 => mean_11
#   x_se_41 => convolution_80
#   x_se_42 => relu_10
#   x_se_43 => convolution_81
# Graph fragment:
#   %sigmoid_63 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_144,), kwargs = {})
#   %mul_248 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_144, %sigmoid_63), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_248, [2, 3], True), kwargs = {})
#   %convolution_80 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_11, %arg129_1, %arg130_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_80,), kwargs = {})
#   %convolution_81 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg131_1, %arg132_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_64 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_81,), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_248, %sigmoid_64), kwargs = {})
triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_37 = async_compile.triton('triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_37(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 65536)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/on/conglid5nl4nuzhdu3id7n5ivrrvd6yfpmimpaj56wuh7m3izgel.py
# Topologically Sorted Source Nodes: [x_230, x_232, x_233, input_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# Source node to ATen node mapping:
#   input_16 => mul_256, sigmoid_65
#   x_230 => add_146, mul_251, mul_252, sub_65
#   x_232 => add_148, mul_254, mul_255, sub_66
#   x_233 => add_149
# Graph fragment:
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_82, %unsqueeze_481), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_483), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_251, %unsqueeze_485), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_252, %unsqueeze_487), kwargs = {})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_83, %unsqueeze_489), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_491), kwargs = {})
#   %mul_255 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_254, %unsqueeze_493), kwargs = {})
#   %add_148 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_255, %unsqueeze_495), kwargs = {})
#   %add_149 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_146, %add_148), kwargs = {})
#   %sigmoid_65 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_149,), kwargs = {})
#   %mul_256 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_149, %sigmoid_65), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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
    tmp30 = tl.sigmoid(tmp29)
    tmp31 = tmp29 * tmp30
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bq/cbqcjxviilgx444qri5e2budah4m2bnrwryou2wt7hfihchiu33g.py
# Topologically Sorted Source Nodes: [x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_235 => add_151, mul_258, mul_259, sub_67
#   x_236 => mul_260, sigmoid_66
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_84, %unsqueeze_497), kwargs = {})
#   %mul_258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_499), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_258, %unsqueeze_501), kwargs = {})
#   %add_151 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_259, %unsqueeze_503), kwargs = {})
#   %sigmoid_66 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_151,), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_151, %sigmoid_66), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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


# kernel path: /tmp/torchinductor_sahanp/rj/crjscm4dgl64rgvb7zcyflufgte2t4rgvb7n3oeon7gn4v7hepav.py
# Topologically Sorted Source Nodes: [x_242, x_243, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# Source node to ATen node mapping:
#   input_17 => mul_269, sigmoid_69
#   x_242 => add_155, mul_267, mul_268, sub_69
#   x_243 => add_156
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_88, %unsqueeze_513), kwargs = {})
#   %mul_267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_515), kwargs = {})
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_267, %unsqueeze_517), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_268, %unsqueeze_519), kwargs = {})
#   %add_156 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_155, %mul_256), kwargs = {})
#   %sigmoid_69 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_156,), kwargs = {})
#   %mul_269 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_156, %sigmoid_69), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_40', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/57/c57jxm35suraazspg62v45lkgqpwiv773wcc4mfktsnfifxzvldd.py
# Topologically Sorted Source Nodes: [reshape_60], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   reshape_60 => clone_36
# Graph fragment:
#   %clone_36 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_15,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_41 = async_compile.triton('triton_poi_fused_clone_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_41(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (196608*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/br/cbrcxt4kp4snwxm5rzbjqmfhyibvb5pulywlmvvewbesrmkmog76.py
# Topologically Sorted Source Nodes: [k_11], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   k_11 => clone_37
# Graph fragment:
#   %clone_37 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_16,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_42 = async_compile.triton('triton_poi_fused_clone_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_42(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (256 + y0 + (768*x2) + (196608*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sc/cscaob6waekgmo3cscp2wc7zovk2qu6pdrgzu3dfmcsp3b4ic2tf.py
# Topologically Sorted Source Nodes: [x_252], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_252 => clone_40
# Graph fragment:
#   %clone_40 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_45,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_43 = async_compile.triton('triton_poi_fused_clone_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_43(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x3 = (xindex // 64)
    y0 = yindex % 16
    y1 = (yindex // 16)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x3) + (256*x2) + (256*((y0 + (16*x3)) // 256)) + (16384*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (1024*y4)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ze/czej3lis4cljbir53s3leqdfhxxjkqqxgebqkngedzg5aq7itf3l.py
# Topologically Sorted Source Nodes: [x_248], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_248 => clone_39
# Graph fragment:
#   %clone_39 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_127,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_44 = async_compile.triton('triton_poi_fused_clone_44', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_44(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (16384*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n2/cn2ghzmgftpinp4j42nuemt3nujotamcoxz2m6xyq27eifgcmmzv.py
# Topologically Sorted Source Nodes: [mul_17, attn_10, attn_11], Original ATen: [aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_10 => add_160
#   attn_11 => amax_5, div_5, exp_5, sub_71, sum_6
#   mul_17 => mul_274
# Graph fragment:
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_10, 0.125), kwargs = {})
#   %add_160 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_274, %view_140), kwargs = {})
#   %amax_5 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_160, [-1], True), kwargs = {})
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_160, %amax_5), kwargs = {})
#   %exp_5 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_71,), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_5, [-1], True), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_5, %sum_6), kwargs = {})
triton_per_fused__softmax_add_mul_45 = async_compile.triton('triton_per_fused__softmax_add_mul_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_mul_45(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), None)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tmp3 = 15 + (31*(x0 // 16)) + (r2 // 16)
    tmp4 = tl.full([1], 512, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
    tmp7 = tl.full([1], 31, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + ((31*(triton_helpers.div_floor_integer(15 + (31*(x0 // 16)) + (r2 // 16),  32))) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = 15 + (31*(x0 % 16)) + (r2 % 16)
    tmp14 = tmp13 < tmp4
    tmp15 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
    tmp16 = tmp15 < tmp7
    tmp17 = tmp16 & tmp14
    tmp18 = tl.load(in_ptr2 + ((31*((15 + (31*(x0 % 16)) + (r2 % 16)) // 32)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), tmp17, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp14, tmp18, tmp19)
    tmp21 = tmp12 + tmp20
    tmp22 = tmp2 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp23, 0))
    tmp26 = tmp22 - tmp25
    tmp27 = tl_math.exp(tmp26)
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = tmp27 / tmp30
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ks/cks4zgmfabrdz7ytnbfsoepsddfuwc3c5xrq5vgqryfuqieoinf4.py
# Topologically Sorted Source Nodes: [reshape_62], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   reshape_62 => clone_38
# Graph fragment:
#   %clone_38 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_17,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_46 = async_compile.triton('triton_poi_fused_clone_46', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_46(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (512 + y0 + (768*x2) + (196608*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cz/cczdht6kqmqi6umz2nsutt5osfhehei7vwf67houmj7r6sbdeucj.py
# Topologically Sorted Source Nodes: [x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_256 => add_162, mul_276, mul_277, sub_72
#   x_257 => mul_278, sigmoid_71
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_144, %unsqueeze_529), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_531), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_276, %unsqueeze_533), kwargs = {})
#   %add_162 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_277, %unsqueeze_535), kwargs = {})
#   %sigmoid_71 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_162,), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_162, %sigmoid_71), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_47', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256) % 256
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (16384*((x1 + (256*x0)) // 16384)) + (65536*x2) + (x0 % 64)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr1 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/u4/cu4nkpihia57qimrxciec2flobdobdhz7causbmdgtckibfy2lza.py
# Topologically Sorted Source Nodes: [x_262, x_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_262 => add_167, mul_284, mul_285, sub_74
#   x_263 => mul_286, sigmoid_73
# Graph fragment:
#   %sub_74 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_92, %unsqueeze_545), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_74, %unsqueeze_547), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_284, %unsqueeze_549), kwargs = {})
#   %add_167 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_285, %unsqueeze_551), kwargs = {})
#   %sigmoid_73 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_167,), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_167, %sigmoid_73), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3v/c3v7ijqlqf46ql5pjpmb3ara5p7w2enkmidpwahflqludeeaqvhj.py
# Topologically Sorted Source Nodes: [reshape_72], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   reshape_72 => clone_43
# Graph fragment:
#   %clone_43 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_18,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_49 = async_compile.triton('triton_poi_fused_clone_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_49(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1536*x2) + (393216*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ig/cigvuf6ucuyev2ivozf7j2ektgzfdiq3fpnxflmrnielwdr2xdgd.py
# Topologically Sorted Source Nodes: [k_13], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   k_13 => clone_44
# Graph fragment:
#   %clone_44 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_19,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_50 = async_compile.triton('triton_poi_fused_clone_50', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_50(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (512 + y0 + (1536*x2) + (393216*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qg/cqgf2wmzdrgg6wuchnxngaayaqt65ph5ag2nztp6wnasyqsetds5.py
# Topologically Sorted Source Nodes: [x_269], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_269 => clone_47
# Graph fragment:
#   %clone_47 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_53,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_51 = async_compile.triton('triton_poi_fused_clone_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_51(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 128
    x3 = (xindex // 128)
    y0 = yindex % 16
    y1 = (yindex // 16)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x3) + (256*x2) + (256*((y0 + (16*x3)) // 256)) + (32768*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (2048*y4)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oo/coo3esoydcvl4mxz5oyvqc7uwhhlbnuid3a5sn2edfnjhxjv4wvs.py
# Topologically Sorted Source Nodes: [x_265], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_265 => clone_46
# Graph fragment:
#   %clone_46 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_151,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_52 = async_compile.triton('triton_poi_fused_clone_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_52(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (32768*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/66/c66zjk6nz42d3ta55j2w4cc3vtjf2ztn7pe2u26rtcxz4kurseer.py
# Topologically Sorted Source Nodes: [mul_18, attn_12, attn_13], Original ATen: [aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_12 => add_169
#   attn_13 => amax_6, div_6, exp_6, sub_75, sum_7
#   mul_18 => mul_287
# Graph fragment:
#   %mul_287 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_12, 0.08838834764831845), kwargs = {})
#   %add_169 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_287, %view_164), kwargs = {})
#   %amax_6 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_169, [-1], True), kwargs = {})
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_169, %amax_6), kwargs = {})
#   %exp_6 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_75,), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_6, [-1], True), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_6, %sum_7), kwargs = {})
triton_per_fused__softmax_add_mul_53 = async_compile.triton('triton_per_fused__softmax_add_mul_53', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_mul_53(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), None)
    tmp1 = 0.08838834764831845
    tmp2 = tmp0 * tmp1
    tmp3 = 15 + (31*(x0 // 16)) + (r2 // 16)
    tmp4 = tl.full([1], 512, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
    tmp7 = tl.full([1], 31, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + ((31*(triton_helpers.div_floor_integer(15 + (31*(x0 // 16)) + (r2 // 16),  32))) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = 15 + (31*(x0 % 16)) + (r2 % 16)
    tmp14 = tmp13 < tmp4
    tmp15 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
    tmp16 = tmp15 < tmp7
    tmp17 = tmp16 & tmp14
    tmp18 = tl.load(in_ptr2 + ((31*((15 + (31*(x0 % 16)) + (r2 % 16)) // 32)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), tmp17, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp14, tmp18, tmp19)
    tmp21 = tmp12 + tmp20
    tmp22 = tmp2 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp23, 0))
    tmp26 = tmp22 - tmp25
    tmp27 = tl_math.exp(tmp26)
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = tmp27 / tmp30
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6w/c6wo32sh5fxonuhljscu6t57wlnoupfudmnc7d5kgtohydvdmgpe.py
# Topologically Sorted Source Nodes: [reshape_74], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   reshape_74 => clone_45
# Graph fragment:
#   %clone_45 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_20,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_54 = async_compile.triton('triton_poi_fused_clone_54', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_54(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (1024 + y0 + (1536*x2) + (393216*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5u/c5um5ee7pctymtucrvdogodhnlyg55y4p6s4d42jd7uyffv2ziv4.py
# Topologically Sorted Source Nodes: [out_8, x_273, x_274], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   out_8 => avg_pool2d_1
#   x_273 => add_171, mul_289, mul_290, sub_76
#   x_274 => mul_291, sigmoid_74
# Graph fragment:
#   %avg_pool2d_1 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%view_168, [2, 2], [2, 2]), kwargs = {})
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_1, %unsqueeze_553), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %unsqueeze_555), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_557), kwargs = {})
#   %add_171 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %unsqueeze_559), kwargs = {})
#   %sigmoid_74 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_171,), kwargs = {})
#   %mul_291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_171, %sigmoid_74), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_silu_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_silu_55', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_silu_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_silu_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512) % 8
    x2 = (xindex // 4096) % 8
    x3 = (xindex // 32768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((256*x1) + (4096*x2) + (32768*((x1 + (16*x2) + (128*x0)) // 16384)) + (131072*x3) + (((x1 + (16*x2) + (128*x0)) // 128) % 128)), None)
    tmp1 = tl.load(in_ptr0 + (128 + (256*x1) + (4096*x2) + (32768*((1 + (2*x1) + (32*x2) + (256*x0)) // 32768)) + (131072*x3) + (((1 + (2*x1) + (32*x2) + (256*x0)) // 256) % 128)), None)
    tmp3 = tl.load(in_ptr0 + (2048 + (256*x1) + (4096*x2) + (32768*((8 + x1 + (16*x2) + (128*x0)) // 16384)) + (131072*x3) + (((8 + x1 + (16*x2) + (128*x0)) // 128) % 128)), None)
    tmp5 = tl.load(in_ptr0 + (2176 + (256*x1) + (4096*x2) + (32768*((17 + (2*x1) + (32*x2) + (256*x0)) // 32768)) + (131072*x3) + (((17 + (2*x1) + (32*x2) + (256*x0)) // 256) % 128)), None)
    tmp9 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.sigmoid(tmp23)
    tmp25 = tmp23 * tmp24
    tl.store(out_ptr1 + (x4), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ji/cji4gbzfrrvob7ehoan46aq2255fnwhe5xzqc26mw5xx6haebh6j.py
# Topologically Sorted Source Nodes: [x_276, x_278, x_279, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# Source node to ATen node mapping:
#   input_19 => mul_298, sigmoid_75
#   x_276 => add_173, mul_293, mul_294, sub_77
#   x_278 => add_175, mul_296, mul_297, sub_78
#   x_279 => add_176
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_94, %unsqueeze_561), kwargs = {})
#   %mul_293 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_563), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_293, %unsqueeze_565), kwargs = {})
#   %add_173 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_294, %unsqueeze_567), kwargs = {})
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_95, %unsqueeze_569), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_571), kwargs = {})
#   %mul_297 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_296, %unsqueeze_573), kwargs = {})
#   %add_175 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_297, %unsqueeze_575), kwargs = {})
#   %add_176 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_173, %add_175), kwargs = {})
#   %sigmoid_75 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_176,), kwargs = {})
#   %mul_298 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_176, %sigmoid_75), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_56', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_56(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1536
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
    tmp30 = tl.sigmoid(tmp29)
    tmp31 = tmp29 * tmp30
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jp/cjpvsur5f5y5juth65wg2kxq33d24grvjuntibjb7cxhe7onsrbb.py
# Topologically Sorted Source Nodes: [x_281, x_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_281 => add_178, mul_300, mul_301, sub_79
#   x_282 => mul_302, sigmoid_76
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_96, %unsqueeze_577), kwargs = {})
#   %mul_300 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_579), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_300, %unsqueeze_581), kwargs = {})
#   %add_178 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_301, %unsqueeze_583), kwargs = {})
#   %sigmoid_76 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_178,), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_178, %sigmoid_76), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_57', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_57', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_57(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7a/c7avqszjljdtmoe6lfwfy5r2ofdpdhlmzz3vp6b2bpacl7oskhkv.py
# Topologically Sorted Source Nodes: [reshape_84], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   reshape_84 => clone_50
# Graph fragment:
#   %clone_50 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_21,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_58 = async_compile.triton('triton_poi_fused_clone_58', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_58(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1536*x2) + (98304*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dp/cdpza3bbsoddrilxb3mmcsy7dqgllxndxxqnj25efmtnqm7uvda7.py
# Topologically Sorted Source Nodes: [k_15], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   k_15 => clone_51
# Graph fragment:
#   %clone_51 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_22,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_59 = async_compile.triton('triton_poi_fused_clone_59', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_59(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (512 + y0 + (1536*x2) + (98304*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qo/cqolepj4l5updumkkfbffqmr7zeepr6chncsto2jbafzjvijc2jt.py
# Topologically Sorted Source Nodes: [x_288], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_288 => clone_54
# Graph fragment:
#   %clone_54 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_61,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_60 = async_compile.triton('triton_poi_fused_clone_60', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_60(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 128
    x3 = (xindex // 128)
    y0 = yindex % 8
    y1 = (yindex // 8)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (8*x3) + (64*x2) + (64*((y0 + (8*x3)) // 64)) + (8192*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (1024*y4)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xs/cxsptvnfuxz4tm54jnz36b3hchb4xe4xatxjdzzstglsoaz2gp5u.py
# Topologically Sorted Source Nodes: [x_284], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_284 => clone_53
# Graph fragment:
#   %clone_53 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_175,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_61 = async_compile.triton('triton_poi_fused_clone_61', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_61(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (8192*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ib/cibbw2rlqjzvkrfcuuh2r3cbakrufyv2ajg2bgylcu62keyd6zdz.py
# Topologically Sorted Source Nodes: [mul_19, attn_14, attn_15], Original ATen: [aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_14 => add_180
#   attn_15 => amax_7, div_7, exp_7, sub_80, sum_8
#   mul_19 => mul_303
# Graph fragment:
#   %mul_303 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_14, 0.08838834764831845), kwargs = {})
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_303, %view_188), kwargs = {})
#   %amax_7 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_180, [-1], True), kwargs = {})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_180, %amax_7), kwargs = {})
#   %exp_7 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_80,), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_7, [-1], True), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_7, %sum_8), kwargs = {})
triton_per_fused__softmax_add_mul_62 = async_compile.triton('triton_per_fused__softmax_add_mul_62', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_mul_62(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), xmask, other=0.0)
    tmp1 = 0.08838834764831845
    tmp2 = tmp0 * tmp1
    tmp3 = 7 + (15*(x0 // 8)) + (r2 // 8)
    tmp4 = tl.full([1, 1], 128, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = (7 + (15*(x0 // 8)) + (r2 // 8)) % 16
    tmp7 = tl.full([1, 1], 15, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + ((15*(triton_helpers.div_floor_integer(7 + (15*(x0 // 8)) + (r2 // 8),  16))) + (120*(x0 % 8)) + (960*x1) + ((7 + (15*(x0 // 8)) + (r2 // 8)) % 16)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = 7 + (15*(x0 % 8)) + (r2 % 8)
    tmp14 = tmp13 < tmp4
    tmp15 = (7 + (15*(x0 % 8)) + (r2 % 8)) % 16
    tmp16 = tmp15 < tmp7
    tmp17 = tmp16 & tmp14
    tmp18 = tl.load(in_ptr2 + ((15*((7 + (15*(x0 % 8)) + (r2 % 8)) // 16)) + (120*(x0 // 8)) + (960*x1) + ((7 + (15*(x0 % 8)) + (r2 % 8)) % 16)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp14, tmp18, tmp19)
    tmp21 = tmp12 + tmp20
    tmp22 = tmp2 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(xmask, tmp23, float("-inf"))
    tmp26 = triton_helpers.max2(tmp25, 1)[:, None]
    tmp27 = tmp22 - tmp26
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp28 / tmp32
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp33, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ki/cki34zratdgk2a6dsca3hwnanqjsnncye2c4g4uoswulhxwf56em.py
# Topologically Sorted Source Nodes: [reshape_86], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   reshape_86 => clone_52
# Graph fragment:
#   %clone_52 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_23,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_63 = async_compile.triton('triton_poi_fused_clone_63', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_63(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (1024 + y0 + (1536*x2) + (98304*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vv/cvve5mtfi3k4xj7bzakv3w74ys5l4hnhhfdenze2quh6fylig4ui.py
# Topologically Sorted Source Nodes: [x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_292 => add_182, mul_305, mul_306, sub_81
#   x_293 => mul_307, sigmoid_77
# Graph fragment:
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_192, %unsqueeze_585), kwargs = {})
#   %mul_305 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_587), kwargs = {})
#   %mul_306 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_305, %unsqueeze_589), kwargs = {})
#   %add_182 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_306, %unsqueeze_591), kwargs = {})
#   %sigmoid_77 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_182,), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_182, %sigmoid_77), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_64 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_64', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512) % 64
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*x1) + (8192*((x1 + (64*x0)) // 8192)) + (32768*x2) + (x0 % 128)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr1 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pq/cpq52ozipgqkxjur47t2nb2y7sy6h5iz6vdmped2adkdjp3yduv2.py
# Topologically Sorted Source Nodes: [x_295, x_296, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# Source node to ATen node mapping:
#   input_20 => mul_311, sigmoid_78
#   x_295 => add_184, mul_309, mul_310, sub_82
#   x_296 => add_185
# Graph fragment:
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_98, %unsqueeze_593), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_82, %unsqueeze_595), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_309, %unsqueeze_597), kwargs = {})
#   %add_184 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_310, %unsqueeze_599), kwargs = {})
#   %add_185 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_184, %mul_298), kwargs = {})
#   %sigmoid_78 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_185,), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_185, %sigmoid_78), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_65 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_silu_65', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_silu_65(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1536
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
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hm/chmmyry45qvmgncdcrum22douvwacvlfcm6k6tyb4mtbyyj7ykaj.py
# Topologically Sorted Source Nodes: [x_298], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_298 => add_187, mul_313, mul_314, sub_83
# Graph fragment:
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_99, %unsqueeze_601), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_603), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_313, %unsqueeze_605), kwargs = {})
#   %add_187 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_314, %unsqueeze_607), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_66 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_66', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_66', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_66(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1280
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


# kernel path: /tmp/torchinductor_sahanp/ei/ceijsv6sa64ny7fbr2s653szdro2qs7aul5wg3r4ncahrucpuohb.py
# Topologically Sorted Source Nodes: [x_299, x_300], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_299 => mul_315, sigmoid_79
#   x_300 => mean_13
# Graph fragment:
#   %sigmoid_79 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_187,), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_187, %sigmoid_79), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_315, [-1, -2], True), kwargs = {})
triton_per_fused_mean_silu_67 = async_compile.triton('triton_per_fused_mean_silu_67', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_67(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (81920*x1)), xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1 = args
    args.clear()
    assert_size_stride(arg0_1, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(arg2_1, (24, ), (1, ))
    assert_size_stride(arg3_1, (24, ), (1, ))
    assert_size_stride(arg4_1, (24, ), (1, ))
    assert_size_stride(arg5_1, (24, ), (1, ))
    assert_size_stride(arg6_1, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (32, ), (1, ))
    assert_size_stride(arg9_1, (32, ), (1, ))
    assert_size_stride(arg10_1, (32, ), (1, ))
    assert_size_stride(arg11_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg27_1, (8, ), (1, ))
    assert_size_stride(arg28_1, (64, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg41_1, (64, ), (1, ))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg46_1, (64, ), (1, ))
    assert_size_stride(arg47_1, (64, ), (1, ))
    assert_size_stride(arg48_1, (64, ), (1, ))
    assert_size_stride(arg49_1, (64, ), (1, ))
    assert_size_stride(arg50_1, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg51_1, (8, ), (1, ))
    assert_size_stride(arg52_1, (64, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg53_1, (64, ), (1, ))
    assert_size_stride(arg54_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg60_1, (128, ), (1, ))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (128, ), (1, ))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg70_1, (8, ), (1, ))
    assert_size_stride(arg71_1, (128, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, ), (1, ))
    assert_size_stride(arg87_1, (128, ), (1, ))
    assert_size_stride(arg88_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg94_1, (8, ), (1, ))
    assert_size_stride(arg95_1, (128, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg96_1, (128, ), (1, ))
    assert_size_stride(arg97_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (512, ), (1, ))
    assert_size_stride(arg101_1, (512, ), (1, ))
    assert_size_stride(arg102_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, ), (1, ))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (384, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg108_1, (63, 32), (32, 1))
    assert_size_stride(arg109_1, (63, 32), (32, 1))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (128, ), (1, ))
    assert_size_stride(arg112_1, (128, ), (1, ))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (512, ), (1, ))
    assert_size_stride(arg117_1, (512, ), (1, ))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg120_1, (256, ), (1, ))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (256, ), (1, ))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (16, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg130_1, (16, ), (1, ))
    assert_size_stride(arg131_1, (256, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg132_1, (256, ), (1, ))
    assert_size_stride(arg133_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (1024, ), (1, ))
    assert_size_stride(arg142_1, (1024, ), (1, ))
    assert_size_stride(arg143_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, ), (1, ))
    assert_size_stride(arg147_1, (256, ), (1, ))
    assert_size_stride(arg148_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (256, ), (1, ))
    assert_size_stride(arg151_1, (256, ), (1, ))
    assert_size_stride(arg152_1, (256, ), (1, ))
    assert_size_stride(arg153_1, (16, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg154_1, (16, ), (1, ))
    assert_size_stride(arg155_1, (256, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg156_1, (256, ), (1, ))
    assert_size_stride(arg157_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg158_1, (1024, ), (1, ))
    assert_size_stride(arg159_1, (1024, ), (1, ))
    assert_size_stride(arg160_1, (1024, ), (1, ))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg163_1, (256, ), (1, ))
    assert_size_stride(arg164_1, (256, ), (1, ))
    assert_size_stride(arg165_1, (256, ), (1, ))
    assert_size_stride(arg166_1, (256, ), (1, ))
    assert_size_stride(arg167_1, (768, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg168_1, (31, 64), (64, 1))
    assert_size_stride(arg169_1, (31, 64), (64, 1))
    assert_size_stride(arg170_1, (256, ), (1, ))
    assert_size_stride(arg171_1, (256, ), (1, ))
    assert_size_stride(arg172_1, (256, ), (1, ))
    assert_size_stride(arg173_1, (256, ), (1, ))
    assert_size_stride(arg174_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (1024, ), (1, ))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg180_1, (512, ), (1, ))
    assert_size_stride(arg181_1, (512, ), (1, ))
    assert_size_stride(arg182_1, (512, ), (1, ))
    assert_size_stride(arg183_1, (512, ), (1, ))
    assert_size_stride(arg184_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg185_1, (31, 128), (128, 1))
    assert_size_stride(arg186_1, (31, 128), (128, 1))
    assert_size_stride(arg187_1, (512, ), (1, ))
    assert_size_stride(arg188_1, (512, ), (1, ))
    assert_size_stride(arg189_1, (512, ), (1, ))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg192_1, (1536, ), (1, ))
    assert_size_stride(arg193_1, (1536, ), (1, ))
    assert_size_stride(arg194_1, (1536, ), (1, ))
    assert_size_stride(arg195_1, (1536, ), (1, ))
    assert_size_stride(arg196_1, (1536, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg197_1, (1536, ), (1, ))
    assert_size_stride(arg198_1, (1536, ), (1, ))
    assert_size_stride(arg199_1, (1536, ), (1, ))
    assert_size_stride(arg200_1, (1536, ), (1, ))
    assert_size_stride(arg201_1, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (512, ), (1, ))
    assert_size_stride(arg204_1, (512, ), (1, ))
    assert_size_stride(arg205_1, (512, ), (1, ))
    assert_size_stride(arg206_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg207_1, (15, 128), (128, 1))
    assert_size_stride(arg208_1, (15, 128), (128, 1))
    assert_size_stride(arg209_1, (512, ), (1, ))
    assert_size_stride(arg210_1, (512, ), (1, ))
    assert_size_stride(arg211_1, (512, ), (1, ))
    assert_size_stride(arg212_1, (512, ), (1, ))
    assert_size_stride(arg213_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg214_1, (1536, ), (1, ))
    assert_size_stride(arg215_1, (1536, ), (1, ))
    assert_size_stride(arg216_1, (1536, ), (1, ))
    assert_size_stride(arg217_1, (1536, ), (1, ))
    assert_size_stride(arg218_1, (1280, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg219_1, (1280, ), (1, ))
    assert_size_stride(arg220_1, (1280, ), (1, ))
    assert_size_stride(arg221_1, (1280, ), (1, ))
    assert_size_stride(arg222_1, (1280, ), (1, ))
    assert_size_stride(arg223_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg224_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 256, 256), (196608, 1, 768, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((24, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 72, 9, grid=grid(72, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 24, 128, 128), (393216, 1, 3072, 24))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((8, 24, 128, 128), (393216, 1, 3072, 24), torch.float32)
        # Topologically Sorted Source Nodes: [x_153, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, buf4, 3145728, grid=grid(3145728), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf3
        buf5 = empty_strided_cuda((32, 24, 3, 3), (216, 1, 72, 24), torch.float32)
        # Topologically Sorted Source Nodes: [x_154, x_155], Original ATen: [aten.silu, aten.convolution]
        triton_poi_fused_convolution_silu_3.run(arg6_1, buf5, 768, 9, grid=grid(768, 9), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [x_154, x_155], Original ATen: [aten.silu, aten.convolution]
        buf6 = extern_kernels.convolution(buf4, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 32, 128, 128), (524288, 1, 4096, 32))
        del buf4
        del buf5
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided_cuda((8, 32, 128, 128), (524288, 1, 4096, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_4.run(buf7, arg7_1, arg8_1, arg9_1, arg10_1, buf8, 4194304, grid=grid(4194304), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        del buf7
        buf9 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_157, x_158], Original ATen: [aten.silu, aten.convolution]
        triton_poi_fused_convolution_silu_5.run(arg11_1, buf9, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [x_157, x_158], Original ATen: [aten.silu, aten.convolution]
        buf10 = extern_kernels.convolution(buf8, buf9, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del buf9
        buf11 = buf10; del buf10  # reuse
        buf12 = empty_strided_cuda((8, 64, 64, 64), (262144, 1, 4096, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_6.run(buf11, arg12_1, arg13_1, arg14_1, arg15_1, buf12, 2097152, grid=grid(2097152), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        # Topologically Sorted Source Nodes: [x_160, x_161], Original ATen: [aten.silu, aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del arg16_1
        buf14 = buf13; del buf13  # reuse
        buf15 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_162, x_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_6.run(buf14, arg17_1, arg18_1, arg19_1, arg20_1, buf15, 2097152, grid=grid(2097152), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del buf14
        buf16 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_163, x_164], Original ATen: [aten.silu, aten.convolution]
        triton_poi_fused_convolution_silu_7.run(arg21_1, buf16, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg21_1
        # Topologically Sorted Source Nodes: [x_163, x_164], Original ATen: [aten.silu, aten.convolution]
        buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del buf15
        buf18 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [x_165], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf18, arg22_1, arg23_1, arg24_1, arg25_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        buf19 = empty_strided_cuda((8, 64, 1, 1, 32), (2048, 1, 16384, 16384, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_166, x_se_24], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_9.run(buf18, buf19, 16384, 128, grid=grid(16384), stream=stream0)
        buf21 = empty_strided_cuda((8, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_166, x_se_24], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_10.run(buf19, buf21, 512, 32, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_166, x_se_24, x_se_25], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg26_1
        del buf21
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_166, x_se_24, x_se_25, x_se_26], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_silu_11.run(buf23, arg27_1, 64, grid=grid(64), stream=stream0)
        del arg27_1
        # Topologically Sorted Source Nodes: [x_166, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
        buf24 = extern_kernels.convolution(buf23, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg28_1
        del buf23
        buf25 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_166, x_se_24, x_se_25, x_se_26, x_se_27, sigmoid_6, x_167], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_12.run(buf25, buf24, arg29_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg29_1
        # Topologically Sorted Source Nodes: [x_166, x_se_24, x_se_25, x_se_26, x_se_27, sigmoid_6, x_167, x_168], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf26 = extern_kernels.convolution(buf25, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 256, 64, 64), (1048576, 1, 16384, 256))
        del arg30_1
        del buf25
        # Topologically Sorted Source Nodes: [x_170], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf12, arg35_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 256, 64, 64), (1048576, 1, 16384, 256))
        del arg35_1
        buf28 = buf26; del buf26  # reuse
        buf29 = empty_strided_cuda((8, 256, 64, 64), (1048576, 1, 16384, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_169, x_171, x_172, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_13.run(buf28, arg31_1, arg32_1, arg33_1, arg34_1, buf27, arg36_1, arg37_1, arg38_1, arg39_1, buf29, 8388608, grid=grid(8388608), stream=stream0)
        del arg31_1
        del arg32_1
        del arg33_1
        del arg34_1
        del arg36_1
        del arg37_1
        del arg38_1
        del arg39_1
        del buf27
        # Topologically Sorted Source Nodes: [input_11, x_173], Original ATen: [aten.silu, aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg40_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del arg40_1
        buf31 = buf30; del buf30  # reuse
        buf32 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_174, x_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_6.run(buf31, arg41_1, arg42_1, arg43_1, arg44_1, buf32, 2097152, grid=grid(2097152), stream=stream0)
        del arg41_1
        del arg42_1
        del arg43_1
        del arg44_1
        del buf31
        buf33 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_175, x_176], Original ATen: [aten.silu, aten.convolution]
        triton_poi_fused_convolution_silu_7.run(arg45_1, buf33, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg45_1
        # Topologically Sorted Source Nodes: [x_175, x_176], Original ATen: [aten.silu, aten.convolution]
        buf34 = extern_kernels.convolution(buf32, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del buf32
        del buf33
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf35, arg46_1, arg47_1, arg48_1, arg49_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg46_1
        del arg47_1
        del arg48_1
        del arg49_1
        buf36 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_178, x_se_28], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_9.run(buf35, buf36, 16384, 128, grid=grid(16384), stream=stream0)
        buf38 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_178, x_se_28], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_10.run(buf36, buf38, 512, 32, grid=grid(512), stream=stream0)
        del buf36
        # Topologically Sorted Source Nodes: [x_178, x_se_28, x_se_29], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf39 = extern_kernels.convolution(buf38, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg50_1
        del buf38
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_178, x_se_28, x_se_29, x_se_30], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_silu_11.run(buf40, arg51_1, 64, grid=grid(64), stream=stream0)
        del arg51_1
        # Topologically Sorted Source Nodes: [x_178, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
        buf41 = extern_kernels.convolution(buf40, arg52_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg52_1
        del buf40
        buf42 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [x_178, x_se_28, x_se_29, x_se_30, x_se_31, sigmoid_7, x_179], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_12.run(buf42, buf41, arg53_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg53_1
        del buf41
        # Topologically Sorted Source Nodes: [x_178, x_se_28, x_se_29, x_se_30, x_se_31, sigmoid_7, x_179, x_180], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf43 = extern_kernels.convolution(buf42, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 256, 64, 64), (1048576, 1, 16384, 256))
        del arg54_1
        buf44 = buf29; del buf29  # reuse
        buf45 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_181, x_182, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf44, buf43, arg55_1, arg56_1, arg57_1, arg58_1, buf45, 8388608, grid=grid(8388608), stream=stream0)
        del arg55_1
        del arg56_1
        del arg57_1
        del arg58_1
        del buf43
        del buf44
        # Topologically Sorted Source Nodes: [input_12, x_183], Original ATen: [aten.silu, aten.convolution]
        buf46 = extern_kernels.convolution(buf45, arg59_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 128, 64, 64), (524288, 1, 8192, 128))
        del arg59_1
        buf47 = buf46; del buf46  # reuse
        buf48 = reinterpret_tensor(buf8, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_184, x_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_15.run(buf47, arg60_1, arg61_1, arg62_1, arg63_1, buf48, 4194304, grid=grid(4194304), stream=stream0)
        del arg60_1
        del arg61_1
        del arg62_1
        del arg63_1
        del buf47
        buf49 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_185, x_186], Original ATen: [aten.silu, aten.convolution]
        triton_poi_fused_convolution_silu_16.run(arg64_1, buf49, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg64_1
        # Topologically Sorted Source Nodes: [x_185, x_186], Original ATen: [aten.silu, aten.convolution]
        buf50 = extern_kernels.convolution(buf48, buf49, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 128, 32, 32), (131072, 1, 4096, 128))
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_187], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_17.run(buf51, arg65_1, arg66_1, arg67_1, arg68_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg65_1
        del arg66_1
        del arg67_1
        del arg68_1
        buf52 = empty_strided_cuda((8, 128, 1, 1, 8), (1024, 1, 8192, 8192, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_188, x_se_32], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_18.run(buf51, buf52, 8192, 128, grid=grid(8192), stream=stream0)
        buf54 = empty_strided_cuda((8, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_188, x_se_32], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_19.run(buf52, buf54, 1024, 8, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_188, x_se_32, x_se_33], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf55 = extern_kernels.convolution(buf54, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg69_1
        del buf54
        buf56 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_188, x_se_32, x_se_33, x_se_34], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_silu_11.run(buf56, arg70_1, 64, grid=grid(64), stream=stream0)
        del arg70_1
        # Topologically Sorted Source Nodes: [x_188, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
        buf57 = extern_kernels.convolution(buf56, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg71_1
        del buf56
        buf58 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_188, x_se_32, x_se_33, x_se_34, x_se_35, sigmoid_8, x_189], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_20.run(buf58, buf57, arg72_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg72_1
        # Topologically Sorted Source Nodes: [x_188, x_se_32, x_se_33, x_se_34, x_se_35, sigmoid_8, x_189, x_190], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf59 = extern_kernels.convolution(buf58, arg73_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 512, 32, 32), (524288, 1, 16384, 512))
        del arg73_1
        # Topologically Sorted Source Nodes: [x_192], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf45, arg78_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 512, 32, 32), (524288, 1, 16384, 512))
        del arg78_1
        del buf45
        buf61 = buf59; del buf59  # reuse
        buf62 = reinterpret_tensor(buf48, (8, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_191, x_193, x_194, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_21.run(buf61, arg74_1, arg75_1, arg76_1, arg77_1, buf60, arg79_1, arg80_1, arg81_1, arg82_1, buf62, 4194304, grid=grid(4194304), stream=stream0)
        del arg74_1
        del arg75_1
        del arg76_1
        del arg77_1
        del arg79_1
        del arg80_1
        del arg81_1
        del arg82_1
        del buf60
        # Topologically Sorted Source Nodes: [input_13, x_195], Original ATen: [aten.silu, aten.convolution]
        buf63 = extern_kernels.convolution(buf62, arg83_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del arg83_1
        buf64 = buf63; del buf63  # reuse
        buf65 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_196, x_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_22.run(buf64, arg84_1, arg85_1, arg86_1, arg87_1, buf65, 1048576, grid=grid(1048576), stream=stream0)
        del arg84_1
        del arg85_1
        del arg86_1
        del arg87_1
        del buf64
        buf66 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [x_197, x_198], Original ATen: [aten.silu, aten.convolution]
        triton_poi_fused_convolution_silu_16.run(arg88_1, buf66, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg88_1
        # Topologically Sorted Source Nodes: [x_197, x_198], Original ATen: [aten.silu, aten.convolution]
        buf67 = extern_kernels.convolution(buf65, buf66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del buf65
        del buf66
        buf68 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_17.run(buf68, arg89_1, arg90_1, arg91_1, arg92_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg89_1
        del arg90_1
        del arg91_1
        del arg92_1
        buf69 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_200, x_se_36], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_18.run(buf68, buf69, 8192, 128, grid=grid(8192), stream=stream0)
        buf71 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_200, x_se_36], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_19.run(buf69, buf71, 1024, 8, grid=grid(1024), stream=stream0)
        del buf69
        # Topologically Sorted Source Nodes: [x_200, x_se_36, x_se_37], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg93_1
        del buf71
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_200, x_se_36, x_se_37, x_se_38], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_silu_11.run(buf73, arg94_1, 64, grid=grid(64), stream=stream0)
        del arg94_1
        # Topologically Sorted Source Nodes: [x_200, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
        buf74 = extern_kernels.convolution(buf73, arg95_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg95_1
        del buf73
        buf75 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_200, x_se_36, x_se_37, x_se_38, x_se_39, sigmoid_9, x_201], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_20.run(buf75, buf74, arg96_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg96_1
        del buf74
        # Topologically Sorted Source Nodes: [x_200, x_se_36, x_se_37, x_se_38, x_se_39, sigmoid_9, x_201, x_202], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf76 = extern_kernels.convolution(buf75, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 512, 32, 32), (524288, 1, 16384, 512))
        del arg97_1
        buf77 = buf62; del buf62  # reuse
        buf78 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_203, x_204, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_23.run(buf77, buf76, arg98_1, arg99_1, arg100_1, arg101_1, buf78, 4194304, grid=grid(4194304), stream=stream0)
        del arg100_1
        del arg101_1
        del arg98_1
        del arg99_1
        del buf76
        # Topologically Sorted Source Nodes: [input_14, x_205], Original ATen: [aten.silu, aten.convolution]
        buf79 = extern_kernels.convolution(buf78, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 128, 32, 32), (131072, 1, 4096, 128))
        del arg102_1
        buf80 = buf79; del buf79  # reuse
        buf81 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_206, x_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_22.run(buf80, arg103_1, arg104_1, arg105_1, arg106_1, buf81, 1048576, grid=grid(1048576), stream=stream0)
        del arg103_1
        del arg104_1
        del arg105_1
        del arg106_1
        # Topologically Sorted Source Nodes: [x_207, x_208], Original ATen: [aten.silu, aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg107_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 384, 32, 32), (393216, 1, 12288, 384))
        del arg107_1
        buf83 = reinterpret_tensor(buf81, (8, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [reshape_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf82, buf83, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf84 = reinterpret_tensor(buf80, (8, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [k_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf82, buf84, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf85 = empty_strided_cuda((32, 1024, 1024), (1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf83, (32, 1024, 32), (32768, 1, 1024), 0), reinterpret_tensor(buf84, (32, 32, 1024), (32768, 1024, 1), 0), out=buf85)
        buf86 = reinterpret_tensor(buf84, (32, 32, 32, 32), (32768, 1024, 32, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_213], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf83, buf86, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        buf87 = empty_strided_cuda((32768, 63), (63, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_213], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (32768, 32), (32, 1), 0), reinterpret_tensor(arg109_1, (32, 63), (1, 32), 0), out=buf87)
        del arg109_1
        buf88 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_209], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf83, buf88, 32768, 32, grid=grid(32768, 32), stream=stream0)
        buf89 = empty_strided_cuda((32768, 63), (63, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_209], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (32768, 32), (32, 1), 0), reinterpret_tensor(arg108_1, (32, 63), (1, 32), 0), out=buf89)
        del arg108_1
        buf92 = empty_strided_cuda((32, 1024, 1024), (1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_14, attn_8, attn_9], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_red_fused__softmax_add_mul_28.run(buf85, buf87, buf89, buf92, 32768, 1024, grid=grid(32768), stream=stream0)
        del buf85
        del buf87
        del buf89
        buf93 = reinterpret_tensor(buf88, (8, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [reshape_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf82, buf93, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del buf82
        buf94 = reinterpret_tensor(buf83, (32, 1024, 32), (32768, 32, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [mul_14, attn_8, attn_9, matmul_19], Original ATen: [aten.mul, aten.add, aten._softmax, aten.bmm]
        extern_kernels.bmm(buf92, reinterpret_tensor(buf93, (32, 1024, 32), (32768, 1, 1024), 0), out=buf94)
        del buf92
        buf96 = reinterpret_tensor(buf93, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_217, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_30.run(buf94, arg110_1, arg111_1, arg112_1, arg113_1, buf96, 1048576, grid=grid(1048576), stream=stream0)
        del arg110_1
        del arg111_1
        del arg112_1
        del arg113_1
        del buf94
        # Topologically Sorted Source Nodes: [x_218, x_219], Original ATen: [aten.silu, aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 512, 32, 32), (524288, 1, 16384, 512))
        del arg114_1
        buf98 = buf78; del buf78  # reuse
        buf99 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [x_220, x_221, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_23.run(buf98, buf97, arg115_1, arg116_1, arg117_1, arg118_1, buf99, 4194304, grid=grid(4194304), stream=stream0)
        del arg115_1
        del arg116_1
        del arg117_1
        del arg118_1
        del buf97
        del buf98
        # Topologically Sorted Source Nodes: [input_15, x_222], Original ATen: [aten.silu, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 256, 32, 32), (262144, 1, 8192, 256))
        del arg119_1
        buf101 = buf100; del buf100  # reuse
        buf102 = reinterpret_tensor(buf42, (8, 256, 32, 32), (262144, 1, 8192, 256), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_223, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_31.run(buf101, arg120_1, arg121_1, arg122_1, arg123_1, buf102, 2097152, grid=grid(2097152), stream=stream0)
        del arg120_1
        del arg121_1
        del arg122_1
        del arg123_1
        del buf101
        buf103 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_224, x_225], Original ATen: [aten.silu, aten.convolution]
        triton_poi_fused_convolution_silu_32.run(arg124_1, buf103, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg124_1
        # Topologically Sorted Source Nodes: [x_224, x_225], Original ATen: [aten.silu, aten.convolution]
        buf104 = extern_kernels.convolution(buf102, buf103, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 256, 16, 16), (65536, 1, 4096, 256))
        buf105 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_226], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf105, arg125_1, arg126_1, arg127_1, arg128_1, 524288, grid=grid(524288), stream=stream0)
        del arg125_1
        del arg126_1
        del arg127_1
        del arg128_1
        buf106 = empty_strided_cuda((8, 256, 1, 1, 2), (512, 1, 4096, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_227, x_se_40], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_34.run(buf105, buf106, 4096, 128, grid=grid(4096), stream=stream0)
        buf108 = empty_strided_cuda((8, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_227, x_se_40], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_35.run(buf106, buf108, 2048, 2, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_227, x_se_40, x_se_41], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf109 = extern_kernels.convolution(buf108, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 16, 1, 1), (16, 1, 1, 1))
        del arg129_1
        del buf108
        buf110 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_227, x_se_40, x_se_41, x_se_42], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_silu_36.run(buf110, arg130_1, 128, grid=grid(128), stream=stream0)
        del arg130_1
        # Topologically Sorted Source Nodes: [x_227, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
        buf111 = extern_kernels.convolution(buf110, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg131_1
        del buf110
        buf112 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_227, x_se_40, x_se_41, x_se_42, x_se_43, sigmoid_10, x_228], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_37.run(buf112, buf111, arg132_1, 524288, grid=grid(524288), stream=stream0)
        del arg132_1
        # Topologically Sorted Source Nodes: [x_227, x_se_40, x_se_41, x_se_42, x_se_43, sigmoid_10, x_228, x_229], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf113 = extern_kernels.convolution(buf112, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
        del arg133_1
        # Topologically Sorted Source Nodes: [x_231], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf99, arg138_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
        del arg138_1
        del buf99
        buf115 = buf113; del buf113  # reuse
        buf116 = reinterpret_tensor(buf102, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_230, x_232, x_233, input_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_38.run(buf115, arg134_1, arg135_1, arg136_1, arg137_1, buf114, arg139_1, arg140_1, arg141_1, arg142_1, buf116, 2097152, grid=grid(2097152), stream=stream0)
        del arg134_1
        del arg135_1
        del arg136_1
        del arg137_1
        del arg139_1
        del arg140_1
        del arg141_1
        del arg142_1
        del buf114
        # Topologically Sorted Source Nodes: [input_16, x_234], Original ATen: [aten.silu, aten.convolution]
        buf117 = extern_kernels.convolution(buf116, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del arg143_1
        buf118 = buf117; del buf117  # reuse
        buf119 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_39.run(buf118, arg144_1, arg145_1, arg146_1, arg147_1, buf119, 524288, grid=grid(524288), stream=stream0)
        del arg144_1
        del arg145_1
        del arg146_1
        del arg147_1
        del buf118
        buf120 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [x_236, x_237], Original ATen: [aten.silu, aten.convolution]
        triton_poi_fused_convolution_silu_32.run(arg148_1, buf120, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg148_1
        # Topologically Sorted Source Nodes: [x_236, x_237], Original ATen: [aten.silu, aten.convolution]
        buf121 = extern_kernels.convolution(buf119, buf120, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del buf119
        del buf120
        buf122 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf122, arg149_1, arg150_1, arg151_1, arg152_1, 524288, grid=grid(524288), stream=stream0)
        del arg149_1
        del arg150_1
        del arg151_1
        del arg152_1
        buf123 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_239, x_se_44], Original ATen: [aten.silu, aten.mean]
        triton_red_fused_mean_silu_34.run(buf122, buf123, 4096, 128, grid=grid(4096), stream=stream0)
        buf125 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [x_239, x_se_44], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_35.run(buf123, buf125, 2048, 2, grid=grid(2048), stream=stream0)
        del buf123
        # Topologically Sorted Source Nodes: [x_239, x_se_44, x_se_45], Original ATen: [aten.silu, aten.mean, aten.convolution]
        buf126 = extern_kernels.convolution(buf125, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 16, 1, 1), (16, 1, 1, 1))
        del arg153_1
        del buf125
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_239, x_se_44, x_se_45, x_se_46], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
        triton_poi_fused_convolution_mean_relu_silu_36.run(buf127, arg154_1, 128, grid=grid(128), stream=stream0)
        del arg154_1
        # Topologically Sorted Source Nodes: [x_239, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu]
        buf128 = extern_kernels.convolution(buf127, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg155_1
        del buf127
        buf129 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_239, x_se_44, x_se_45, x_se_46, x_se_47, sigmoid_11, x_240], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        triton_poi_fused_convolution_mean_mul_relu_sigmoid_silu_37.run(buf129, buf128, arg156_1, 524288, grid=grid(524288), stream=stream0)
        del arg156_1
        del buf128
        # Topologically Sorted Source Nodes: [x_239, x_se_44, x_se_45, x_se_46, x_se_47, sigmoid_11, x_240, x_241], Original ATen: [aten.silu, aten.mean, aten.convolution, aten.relu, aten.sigmoid, aten.mul]
        buf130 = extern_kernels.convolution(buf129, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
        del arg157_1
        buf131 = buf116; del buf116  # reuse
        buf132 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_242, x_243, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_40.run(buf131, buf130, arg158_1, arg159_1, arg160_1, arg161_1, buf132, 2097152, grid=grid(2097152), stream=stream0)
        del arg158_1
        del arg159_1
        del arg160_1
        del arg161_1
        # Topologically Sorted Source Nodes: [input_17, x_244], Original ATen: [aten.silu, aten.convolution]
        buf133 = extern_kernels.convolution(buf132, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 256, 16, 16), (65536, 1, 4096, 256))
        del arg162_1
        buf134 = buf133; del buf133  # reuse
        buf135 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [x_245, x_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_39.run(buf134, arg163_1, arg164_1, arg165_1, arg166_1, buf135, 524288, grid=grid(524288), stream=stream0)
        del arg163_1
        del arg164_1
        del arg165_1
        del arg166_1
        # Topologically Sorted Source Nodes: [x_246, x_247], Original ATen: [aten.silu, aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 768, 16, 16), (196608, 1, 12288, 768))
        del arg167_1
        buf137 = reinterpret_tensor(buf135, (8, 256, 16, 16), (65536, 256, 16, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [reshape_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf136, buf137, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf138 = reinterpret_tensor(buf134, (8, 256, 16, 16), (65536, 256, 16, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [k_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_42.run(buf136, buf138, 2048, 256, grid=grid(2048, 256), stream=stream0)
        buf139 = reinterpret_tensor(buf131, (32, 256, 256), (65536, 256, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (32, 256, 64), (16384, 1, 256), 0), reinterpret_tensor(buf138, (32, 64, 256), (16384, 256, 1), 0), out=buf139)
        buf140 = reinterpret_tensor(buf138, (32, 16, 16, 64), (16384, 1024, 64, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [x_252], Original ATen: [aten.clone]
        triton_poi_fused_clone_43.run(buf137, buf140, 512, 1024, grid=grid(512, 1024), stream=stream0)
        buf141 = empty_strided_cuda((8192, 31), (31, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_252], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (8192, 64), (64, 1), 0), reinterpret_tensor(arg169_1, (64, 31), (1, 64), 0), out=buf141)
        del arg169_1
        buf142 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_248], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf137, buf142, 8192, 64, grid=grid(8192, 64), stream=stream0)
        buf143 = empty_strided_cuda((8192, 31), (31, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_248], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf142, (8192, 64), (64, 1), 0), reinterpret_tensor(arg168_1, (64, 31), (1, 64), 0), out=buf143)
        del arg168_1
        buf146 = reinterpret_tensor(buf130, (32, 256, 256), (65536, 256, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [mul_17, attn_10, attn_11], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_45.run(buf139, buf141, buf143, buf146, 8192, 256, grid=grid(8192), stream=stream0)
        del buf139
        buf147 = reinterpret_tensor(buf142, (8, 256, 16, 16), (65536, 256, 16, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [reshape_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_46.run(buf136, buf147, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf136
        buf148 = reinterpret_tensor(buf137, (32, 256, 64), (16384, 64, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [mul_17, attn_10, attn_11, matmul_23], Original ATen: [aten.mul, aten.add, aten._softmax, aten.bmm]
        extern_kernels.bmm(buf146, reinterpret_tensor(buf147, (32, 256, 64), (16384, 1, 256), 0), out=buf148)
        buf150 = reinterpret_tensor(buf147, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_47.run(buf148, arg170_1, arg171_1, arg172_1, arg173_1, buf150, 524288, grid=grid(524288), stream=stream0)
        del arg170_1
        del arg171_1
        del arg172_1
        del arg173_1
        del buf148
        # Topologically Sorted Source Nodes: [x_257, x_258], Original ATen: [aten.silu, aten.convolution]
        buf151 = extern_kernels.convolution(buf150, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
        del arg174_1
        del buf150
        buf152 = buf132; del buf132  # reuse
        buf153 = reinterpret_tensor(buf146, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_259, x_260, input_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_40.run(buf152, buf151, arg175_1, arg176_1, arg177_1, arg178_1, buf153, 2097152, grid=grid(2097152), stream=stream0)
        del arg175_1
        del arg176_1
        del arg177_1
        del arg178_1
        # Topologically Sorted Source Nodes: [input_18, x_261], Original ATen: [aten.silu, aten.convolution]
        buf154 = extern_kernels.convolution(buf153, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 512, 16, 16), (131072, 1, 8192, 512))
        del arg179_1
        buf155 = buf154; del buf154  # reuse
        buf156 = reinterpret_tensor(buf96, (8, 512, 16, 16), (131072, 1, 8192, 512), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_262, x_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_48.run(buf155, arg180_1, arg181_1, arg182_1, arg183_1, buf156, 1048576, grid=grid(1048576), stream=stream0)
        del arg180_1
        del arg181_1
        del arg182_1
        del arg183_1
        # Topologically Sorted Source Nodes: [x_263, x_264], Original ATen: [aten.silu, aten.convolution]
        buf157 = extern_kernels.convolution(buf156, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 1536, 16, 16), (393216, 1, 24576, 1536))
        del arg184_1
        buf158 = reinterpret_tensor(buf156, (8, 512, 16, 16), (131072, 256, 16, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [reshape_72], Original ATen: [aten.clone]
        triton_poi_fused_clone_49.run(buf157, buf158, 4096, 256, grid=grid(4096, 256), stream=stream0)
        buf159 = reinterpret_tensor(buf155, (8, 512, 16, 16), (131072, 256, 16, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [k_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_50.run(buf157, buf159, 4096, 256, grid=grid(4096, 256), stream=stream0)
        buf160 = reinterpret_tensor(buf152, (32, 256, 256), (65536, 256, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf158, (32, 256, 128), (32768, 1, 256), 0), reinterpret_tensor(buf159, (32, 128, 256), (32768, 256, 1), 0), out=buf160)
        buf161 = reinterpret_tensor(buf159, (32, 16, 16, 128), (32768, 2048, 128, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_269], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(buf158, buf161, 512, 2048, grid=grid(512, 2048), stream=stream0)
        buf162 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_269], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (8192, 128), (128, 1), 0), reinterpret_tensor(arg186_1, (128, 31), (1, 128), 0), out=buf162)
        del arg186_1
        buf163 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_265], Original ATen: [aten.clone]
        triton_poi_fused_clone_52.run(buf158, buf163, 8192, 128, grid=grid(8192, 128), stream=stream0)
        buf164 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [x_265], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (8192, 128), (128, 1), 0), reinterpret_tensor(arg185_1, (128, 31), (1, 128), 0), out=buf164)
        del arg185_1
        buf167 = reinterpret_tensor(buf151, (32, 256, 256), (65536, 256, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [mul_18, attn_12, attn_13], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_53.run(buf160, buf162, buf164, buf167, 8192, 256, grid=grid(8192), stream=stream0)
        del buf160
        del buf162
        del buf164
        buf168 = reinterpret_tensor(buf163, (8, 512, 16, 16), (131072, 256, 16, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [reshape_74], Original ATen: [aten.clone]
        triton_poi_fused_clone_54.run(buf157, buf168, 4096, 256, grid=grid(4096, 256), stream=stream0)
        del buf157
        buf169 = reinterpret_tensor(buf158, (32, 256, 128), (32768, 128, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [mul_18, attn_12, attn_13, matmul_27], Original ATen: [aten.mul, aten.add, aten._softmax, aten.bmm]
        extern_kernels.bmm(buf167, reinterpret_tensor(buf168, (32, 256, 128), (32768, 1, 256), 0), out=buf169)
        del buf167
        del buf168
        buf171 = empty_strided_cuda((8, 512, 8, 8), (32768, 1, 4096, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_8, x_273, x_274], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_silu_55.run(buf169, arg187_1, arg188_1, arg189_1, arg190_1, buf171, 262144, grid=grid(262144), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        del buf169
        # Topologically Sorted Source Nodes: [x_274, x_275], Original ATen: [aten.silu, aten.convolution]
        buf172 = extern_kernels.convolution(buf171, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
        del arg191_1
        # Topologically Sorted Source Nodes: [x_277], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf153, arg196_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
        del arg196_1
        del buf153
        buf174 = buf172; del buf172  # reuse
        buf175 = empty_strided_cuda((8, 1536, 8, 8), (98304, 1, 12288, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [x_276, x_278, x_279, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_56.run(buf174, arg192_1, arg193_1, arg194_1, arg195_1, buf173, arg197_1, arg198_1, arg199_1, arg200_1, buf175, 786432, grid=grid(786432), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        del buf173
        del buf174
        # Topologically Sorted Source Nodes: [input_19, x_280], Original ATen: [aten.silu, aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 512, 8, 8), (32768, 1, 4096, 512))
        del arg201_1
        buf177 = buf176; del buf176  # reuse
        buf178 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_281, x_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_57.run(buf177, arg202_1, arg203_1, arg204_1, arg205_1, buf178, 262144, grid=grid(262144), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        # Topologically Sorted Source Nodes: [x_282, x_283], Original ATen: [aten.silu, aten.convolution]
        buf179 = extern_kernels.convolution(buf178, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
        del arg206_1
        buf180 = reinterpret_tensor(buf178, (8, 512, 8, 8), (32768, 64, 8, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [reshape_84], Original ATen: [aten.clone]
        triton_poi_fused_clone_58.run(buf179, buf180, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf181 = reinterpret_tensor(buf177, (8, 512, 8, 8), (32768, 64, 8, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [k_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_59.run(buf179, buf181, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf182 = empty_strided_cuda((32, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (32, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf181, (32, 128, 64), (8192, 64, 1), 0), out=buf182)
        buf183 = reinterpret_tensor(buf181, (32, 8, 8, 128), (8192, 1024, 128, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [x_288], Original ATen: [aten.clone]
        triton_poi_fused_clone_60.run(buf180, buf183, 256, 1024, grid=grid(256, 1024), stream=stream0)
        buf184 = empty_strided_cuda((2048, 15), (15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_288], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (2048, 128), (128, 1), 0), reinterpret_tensor(arg208_1, (128, 15), (1, 128), 0), out=buf184)
        del arg208_1
        buf185 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [x_284], Original ATen: [aten.clone]
        triton_poi_fused_clone_61.run(buf180, buf185, 2048, 128, grid=grid(2048, 128), stream=stream0)
        buf186 = empty_strided_cuda((2048, 15), (15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_284], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf185, (2048, 128), (128, 1), 0), reinterpret_tensor(arg207_1, (128, 15), (1, 128), 0), out=buf186)
        del arg207_1
        buf189 = empty_strided_cuda((32, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_19, attn_14, attn_15], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_62.run(buf182, buf184, buf186, buf189, 2048, 64, grid=grid(2048), stream=stream0)
        del buf182
        del buf184
        del buf186
        buf190 = reinterpret_tensor(buf185, (8, 512, 8, 8), (32768, 64, 8, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [reshape_86], Original ATen: [aten.clone]
        triton_poi_fused_clone_63.run(buf179, buf190, 4096, 64, grid=grid(4096, 64), stream=stream0)
        buf191 = reinterpret_tensor(buf180, (32, 64, 128), (8192, 128, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [mul_19, attn_14, attn_15, matmul_31], Original ATen: [aten.mul, aten.add, aten._softmax, aten.bmm]
        extern_kernels.bmm(buf189, reinterpret_tensor(buf190, (32, 64, 128), (8192, 1, 64), 0), out=buf191)
        del buf189
        buf193 = reinterpret_tensor(buf190, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_64.run(buf191, arg209_1, arg210_1, arg211_1, arg212_1, buf193, 262144, grid=grid(262144), stream=stream0)
        del arg209_1
        del arg210_1
        del arg211_1
        del arg212_1
        del buf191
        # Topologically Sorted Source Nodes: [x_293, x_294], Original ATen: [aten.silu, aten.convolution]
        buf194 = extern_kernels.convolution(buf193, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
        del arg213_1
        del buf193
        buf195 = buf175; del buf175  # reuse
        buf196 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_295, x_296, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_65.run(buf195, buf194, arg214_1, arg215_1, arg216_1, arg217_1, buf196, 786432, grid=grid(786432), stream=stream0)
        del arg214_1
        del arg215_1
        del arg216_1
        del arg217_1
        del buf194
        del buf195
        # Topologically Sorted Source Nodes: [input_20, x_297], Original ATen: [aten.silu, aten.convolution]
        buf197 = extern_kernels.convolution(buf196, arg218_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 1280, 8, 8), (81920, 1, 10240, 1280))
        del arg218_1
        del buf196
        buf198 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [x_298], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_66.run(buf198, arg219_1, arg220_1, arg221_1, arg222_1, 655360, grid=grid(655360), stream=stream0)
        del arg219_1
        del arg220_1
        del arg221_1
        del arg222_1
        buf200 = empty_strided_cuda((8, 1280, 1, 1), (1280, 1, 10240, 10240), torch.float32)
        # Topologically Sorted Source Nodes: [x_299, x_300], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_67.run(buf198, buf200, 10240, 64, grid=grid(10240), stream=stream0)
        del buf198
        buf201 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_303], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg224_1, reinterpret_tensor(buf200, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg223_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf201)
        del arg223_1
        del arg224_1
        del buf200
    return (buf201, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((64, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((63, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((63, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((16, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((16, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((256, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((31, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((31, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((31, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((31, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1536, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((15, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((15, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1280, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('sebotnet33ts_256', benchmark_compiled_module)
