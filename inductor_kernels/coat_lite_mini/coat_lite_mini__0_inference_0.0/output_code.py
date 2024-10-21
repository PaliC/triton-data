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
# Topologically Sorted Source Nodes: [x_148], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_148 => convolution_36
# Graph fragment:
#   %convolution_36 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/dk/cdkrxm2oguw66kqtc67klvcxub3xn6mx22aiioll6anadn73qikk.py
# Topologically Sorted Source Nodes: [x_148], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_148 => convolution_36
# Graph fragment:
#   %convolution_36 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, %arg2_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (48*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4o/c4oyvpe4thkymfl5td6qqenkvdblqpg76ki5nnm46f7penl4elcu.py
# Topologically Sorted Source Nodes: [x_150], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_150 => add_82, add_83, clone_65, mul_82, mul_83, rsqrt_21, sub_29, var_mean_21
# Graph fragment:
#   %clone_65 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_97,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_65, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_65, %getitem_91), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_90, 1e-05), kwargs = {})
#   %rsqrt_21 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_82,), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %rsqrt_21), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %arg3_1), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %arg4_1), kwargs = {})
triton_per_fused_native_layer_norm_2 = async_compile.triton('triton_per_fused_native_layer_norm_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 3136
    x3 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 64.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r1 + (64*x2) + (200768*x3)), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yj/cyj5lswpozs46rnjzxeoltcv5sdyztihys4ngkrqdwfh7rifvvp7.py
# Topologically Sorted Source Nodes: [x_151], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_151 => cat_20
# Graph fragment:
#   %cat_20 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_36, %add_83], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0 + (200768*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/le/cle4no4cyfrugz62mesaexttdegti3tc45c4m4lq55pdeuoa26mr.py
# Topologically Sorted Source Nodes: [x_154, x_155], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_154 => cat_21
#   x_155 => add_85, add_86, mul_84, mul_85, rsqrt_22, sub_30, var_mean_22
# Graph fragment:
#   %cat_21 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_111, %permute_99], 1), kwargs = {})
#   %var_mean_22 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_21, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_21, %getitem_93), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_92, 1e-06), kwargs = {})
#   %rsqrt_22 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_85,), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %rsqrt_22), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %arg8_1), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_85, %arg9_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_4 = async_compile.triton('triton_per_fused_cat_native_layer_norm_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 3137
    r2 = rindex
    x1 = (xindex // 3137)
    x3 = xindex
    tmp40 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (200768*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 3137, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (64*(((-1) + x0) % 3136)) + (200704*x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (64 + r2 + (64*(((-1) + x0) % 3136)) + (200768*x1)), tmp6 & xmask, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp16 - tmp26
    tmp34 = 64.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr3 + (r2 + (64*x3)), tmp43, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/on/con5zxcc4q5setzcq4gmiy7n4gz25pxdtpcg6cs7h6ntvwvj4fob.py
# Topologically Sorted Source Nodes: [k_softmax_8], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_8 => amax_8, clone_66
# Graph fragment:
#   %clone_66 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_95,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_8 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_66, [2], True), kwargs = {})
triton_red_fused__softmax_5 = async_compile.triton('triton_red_fused__softmax_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_5(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64) % 25
    x0 = xindex % 64
    x2 = (xindex // 1600)
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3137, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (64 + x0 + (192*r3) + (24192*x1) + (602304*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/34/c34qe7576z3rxbtnvtcs4suh4wzb4zfam7hbzxzsca6quzb257mx.py
# Topologically Sorted Source Nodes: [k_softmax_8], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_8 => amax_8, clone_66
# Graph fragment:
#   %clone_66 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_95,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_8 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_66, [2], True), kwargs = {})
triton_per_fused__softmax_6 = async_compile.triton('triton_per_fused__softmax_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_6(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (1600*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vw/cvweqrqzbyj6npspifeecxcmzekan6g5txvglk6oqnfi6da24tts.py
# Topologically Sorted Source Nodes: [k_softmax_8], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_8 => clone_66, exp_8, sub_31, sum_9
# Graph fragment:
#   %clone_66 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_95,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_66, %amax_8), kwargs = {})
#   %exp_8 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_31,), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_8, [2], True), kwargs = {})
triton_red_fused__softmax_7 = async_compile.triton('triton_red_fused__softmax_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_7(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64) % 25
    x0 = xindex % 64
    x2 = (xindex // 1600)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3137, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (64 + x0 + (192*r3) + (24192*x1) + (602304*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0 + (64*x2), [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7j/c7jy7swlrsv6ldukhaaidot2zrtpcygfepmsa3kiqvjt3dxqrsew.py
# Topologically Sorted Source Nodes: [k_softmax_8], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_8 => clone_66, exp_8, sub_31, sum_9
# Graph fragment:
#   %clone_66 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_95,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_66, %amax_8), kwargs = {})
#   %exp_8 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_31,), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_8, [2], True), kwargs = {})
triton_per_fused__softmax_8 = async_compile.triton('triton_per_fused__softmax_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_8(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (1600*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/w2/cw2cihsouss2jsrmuyeklvgnia5eapm5po7ggsqtksgzepwvo2c3.py
# Topologically Sorted Source Nodes: [k_softmax_8], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_8 => clone_66, div_8, exp_8, sub_31
# Graph fragment:
#   %clone_66 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_95,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_66, %amax_8), kwargs = {})
#   %exp_8 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_31,), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_8, %sum_9), kwargs = {})
triton_poi_fused__softmax_9 = async_compile.triton('triton_poi_fused__softmax_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 3137
    x2 = (xindex // 25096) % 8
    x3 = (xindex // 200768)
    x4 = (xindex // 25096)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (64 + x0 + (8*x2) + (192*x1) + (602304*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (8*x4)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 + (8*x4)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ly/clyve6ios4lf6bo47v3niveq5ful3umbo5zk32fjq2s4cqv4b7vr.py
# Topologically Sorted Source Nodes: [factor_att_16], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   factor_att_16 => clone_67
# Graph fragment:
#   %clone_67 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_38,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_10 = async_compile.triton('triton_poi_fused_clone_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 3137
    x2 = (xindex // 25096) % 8
    x3 = (xindex // 200768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (8*x2) + (192*x1) + (602304*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/m4/cm422kluymf64rflepusmxrfantiqtnbizukjdlybdjdkfchgljt.py
# Topologically Sorted Source Nodes: [factor_att_17], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   factor_att_17 => clone_68
# Graph fragment:
#   %clone_68 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_39,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_11 = async_compile.triton('triton_poi_fused_clone_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 3137
    x2 = (xindex // 25096) % 8
    x3 = (xindex // 200768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8*x2) + (192*x1) + (602304*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tf/ctfj7xvsfj7lvhsdhpgguigrdtm6km4cj6ht6nwwps3bzjqh6rlf.py
# Topologically Sorted Source Nodes: [EV_hat_16, EV_hat_17, x_157], Original ATen: [aten.mul, aten.constant_pad_nd, aten.clone]
# Source node to ATen node mapping:
#   EV_hat_16 => mul_86
#   EV_hat_17 => constant_pad_nd_8
#   x_157 => clone_69
# Graph fragment:
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_116, %permute_104), kwargs = {})
#   %constant_pad_nd_8 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_86, [0, 0, 1, 0, 0, 0], 0.0), kwargs = {})
#   %clone_69 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_105,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_constant_pad_nd_mul_12 = async_compile.triton('triton_poi_fused_clone_constant_pad_nd_mul_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_constant_pad_nd_mul_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_constant_pad_nd_mul_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 64) % 3137
    x4 = xindex % 64
    x5 = (xindex // 64)
    x0 = xindex % 8
    x1 = (xindex // 8) % 8
    x3 = (xindex // 200768)
    x6 = xindex
    tmp38 = tl.load(in_ptr7 + (x0 + (8*x2) + (25096*x1) + (200768*x3)), xmask)
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x4 + (192*x5)), tmp2 & xmask, other=0.0)
    tmp4 = x4
    tmp5 = tmp4 >= tmp1
    tmp6 = tl.full([1], 16, tl.int64)
    tmp7 = tmp4 < tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = tl.load(in_ptr1 + ((16*(((-1) + x2) % 3136)) + (50176*x3) + (x0 + (8*x1))), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0 + (8*x1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tmp4 >= tmp6
    tmp15 = tl.full([1], 40, tl.int64)
    tmp16 = tmp4 < tmp15
    tmp17 = tmp14 & tmp16
    tmp18 = tmp17 & tmp2
    tmp19 = tl.load(in_ptr3 + ((24*(((-1) + x2) % 3136)) + (75264*x3) + ((-16) + x0 + (8*x1))), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr4 + ((-16) + x0 + (8*x1)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp18, tmp21, tmp22)
    tmp24 = tmp4 >= tmp15
    tmp25 = tl.full([1], 64, tl.int64)
    tmp26 = tmp4 < tmp25
    tmp27 = tmp24 & tmp2
    tmp28 = tl.load(in_ptr5 + ((24*(((-1) + x2) % 3136)) + (75264*x3) + ((-40) + x0 + (8*x1))), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-40) + x0 + (8*x1)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp27, tmp30, tmp31)
    tmp33 = tl.where(tmp17, tmp23, tmp32)
    tmp34 = tl.where(tmp7, tmp13, tmp33)
    tmp35 = tmp3 * tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp2, tmp35, tmp36)
    tmp39 = 0.3535533905932738
    tmp40 = tmp38 * tmp39
    tmp41 = tmp40 + tmp37
    tl.store(out_ptr1 + (x6), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/aw/cawcaqh545tzv4hijcyf4tz6trjj6lprc3cgv2bfhbdabdacin3f.py
# Topologically Sorted Source Nodes: [x_154, x_160, x_161], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_154 => cat_21
#   x_160 => add_88
#   x_161 => add_89, add_90, mul_88, mul_89, rsqrt_23, sub_32, var_mean_23
# Graph fragment:
#   %cat_21 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_111, %permute_99], 1), kwargs = {})
#   %add_88 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_21, %view_184), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_88, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_88, %getitem_101), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_100, 1e-06), kwargs = {})
#   %rsqrt_23 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_89,), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %rsqrt_23), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %arg20_1), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %arg21_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_13 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 3137
    r2 = rindex
    x1 = (xindex // 3137)
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (r2 + (64*x3)), xmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (200768*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 3137, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (64*(((-1) + x0) % 3136)) + (200704*x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (64 + r2 + (64*(((-1) + x0) % 3136)) + (200768*x1)), tmp6 & xmask, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp20 - tmp30
    tmp38 = 64.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp20, xmask)
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp47, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h2/ch2lhoukut4q3ln2xs3k74hv6v4e3rq7lisr4oqjnap3tisifzjw.py
# Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_163 => add_91, erf_8, mul_90, mul_91, mul_92
# Graph fragment:
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_186, 0.5), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_186, 0.7071067811865476), kwargs = {})
#   %erf_8 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_91,), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_8, 1), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %add_91), kwargs = {})
triton_poi_fused_gelu_14 = async_compile.triton('triton_poi_fused_gelu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12849152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
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


# kernel path: /tmp/torchinductor_sahanp/ik/cikgdeyitzsdwa3uwmlt4oaklfoiocvjy2enqt3ysa2mwxyjmmyw.py
# Topologically Sorted Source Nodes: [x_167], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_167 => add_92
# Graph fragment:
#   %add_92 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %view_188), kwargs = {})
triton_poi_fused_add_15 = async_compile.triton('triton_poi_fused_add_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_15(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qm/cqm5ztfrc655uqsigv635m7ikh4u6rywotnhq7eayzvv627nl5m3.py
# Topologically Sorted Source Nodes: [x_170, x_171], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_170 => cat_23
#   x_171 => add_94, add_95, mul_93, mul_94, rsqrt_24, sub_33, var_mean_24
# Graph fragment:
#   %cat_23 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_123, %permute_110], 1), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_23, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_23, %getitem_103), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_102, 1e-06), kwargs = {})
#   %rsqrt_24 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_94,), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %rsqrt_24), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, %arg26_1), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %arg27_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_16 = async_compile.triton('triton_per_fused_cat_native_layer_norm_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 3137
    r2 = rindex
    x1 = (xindex // 3137)
    x3 = xindex
    tmp40 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (64*x0) + (200768*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 3137, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (64*(((-1) + x0) % 3136)) + (200704*x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (64 + r2 + (64*(((-1) + x0) % 3136)) + (200768*x1)), tmp6 & xmask, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp16 - tmp26
    tmp34 = 64.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr3 + (r2 + (64*x3)), tmp43, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yb/cybzmjcckbceg562pvdoe3f6xcgpjr4ei4o5hqselqm4bc2et66q.py
# Topologically Sorted Source Nodes: [x_170, x_176, x_177], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_170 => cat_23
#   x_176 => add_97
#   x_177 => add_98, add_99, mul_97, mul_98, rsqrt_25, sub_35, var_mean_25
# Graph fragment:
#   %cat_23 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_123, %permute_110], 1), kwargs = {})
#   %add_97 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_23, %view_204), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_97, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_97, %getitem_111), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_110, 1e-06), kwargs = {})
#   %rsqrt_25 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_98,), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %rsqrt_25), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %arg32_1), kwargs = {})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %arg33_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_17 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 3137
    r2 = rindex
    x1 = (xindex // 3137)
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (r2 + (64*x3)), xmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (64*x0) + (200768*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 3137, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (64*(((-1) + x0) % 3136)) + (200704*x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (64 + r2 + (64*(((-1) + x0) % 3136)) + (200768*x1)), tmp6 & xmask, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp20 - tmp30
    tmp38 = 64.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp20, xmask)
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp47, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/76/c76safvkhyx645h7gnvhbabdxjrnhoxbqavqcsrrlb4qloobduya.py
# Topologically Sorted Source Nodes: [x1_nocls_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x1_nocls_1 => clone_80
# Graph fragment:
#   %clone_80 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_120,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_18 = async_compile.triton('triton_poi_fused_clone_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_18(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 200704)
    x3 = xindex % 200704
    x0 = xindex % 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (64 + x3 + (200768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (64 + x3 + (200768*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/v2/cv2gsfdk6smjtdseiqk6vsmu2sptv5n6ovi67szgft474zj2bqyu.py
# Topologically Sorted Source Nodes: [x1_nocls_1, x_184], Original ATen: [aten.clone, aten.convolution]
# Source node to ATen node mapping:
#   x1_nocls_1 => clone_80
#   x_184 => convolution_45
# Graph fragment:
#   %clone_80 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_120,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_45 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_80, %arg38_1, %arg39_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_clone_convolution_19 = async_compile.triton('triton_poi_fused_clone_convolution_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (256*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bl/cblemikyztqoheyraay3k3xjqqpbx3axwo63c2vhgd4n5blzbwsl.py
# Topologically Sorted Source Nodes: [x_186], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_186 => add_102, add_103, clone_81, mul_102, mul_103, rsqrt_26, sub_36, var_mean_26
# Graph fragment:
#   %clone_81 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_121,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_81, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_81, %getitem_113), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_112, 1e-05), kwargs = {})
#   %rsqrt_26 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_102,), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %rsqrt_26), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, %arg40_1), kwargs = {})
#   %add_103 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_103, %arg41_1), kwargs = {})
triton_per_fused_native_layer_norm_20 = async_compile.triton('triton_per_fused_native_layer_norm_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 784
    x3 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 128.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r1 + (128*x2) + (100480*x3)), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7h/c7hbug2b6fzsn6oartzbf4c6xi66mw3ofy6qrz6xf5zzi26naul3.py
# Topologically Sorted Source Nodes: [x_187], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_187 => cat_25
# Graph fragment:
#   %cat_25 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_45, %add_103], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0 + (100480*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lu/cluzt3s7tx2g4tod33obxnt6znfmqk5dbh4ansbs2euq4wjbvm7r.py
# Topologically Sorted Source Nodes: [x_190, x_191], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_190 => cat_26
#   x_191 => add_105, add_106, mul_104, mul_105, rsqrt_27, sub_37, var_mean_27
# Graph fragment:
#   %cat_26 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_138, %permute_123], 1), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_26, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_26, %getitem_115), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_114, 1e-06), kwargs = {})
#   %rsqrt_27 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_105,), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %rsqrt_27), kwargs = {})
#   %mul_105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_104, %arg45_1), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_105, %arg46_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_22 = async_compile.triton('triton_per_fused_cat_native_layer_norm_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 785
    r2 = rindex
    x1 = (xindex // 785)
    x3 = xindex
    tmp40 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (100480*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 785, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (128*(((-1) + x0) % 784)) + (100352*x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (128 + r2 + (128*(((-1) + x0) % 784)) + (100480*x1)), tmp6 & xmask, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp16 - tmp26
    tmp34 = 128.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp43, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cl/cclar4abemp3trjpezfk5xhtulfohguz6pkvqilmjppu7enrnju3.py
# Topologically Sorted Source Nodes: [k_softmax_10], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_10 => amax_10, clone_82
# Graph fragment:
#   %clone_82 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_117,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_10 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_82, [2], True), kwargs = {})
triton_red_fused__softmax_23 = async_compile.triton('triton_red_fused__softmax_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_23(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7168
    rnumel = 113
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128) % 7
    x0 = xindex % 128
    x2 = (xindex // 896)
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (113*x1)
        tmp1 = tl.full([1, 1], 785, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (128 + x0 + (384*r3) + (43392*x1) + (301440*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cf/ccfxbog5avmwh745jgnmqhpcfwiyth4yq676k7l5l7c5yw6sfc7v.py
# Topologically Sorted Source Nodes: [k_softmax_10], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_10 => amax_10, clone_82
# Graph fragment:
#   %clone_82 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_117,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_10 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_82, [2], True), kwargs = {})
triton_per_fused__softmax_24 = async_compile.triton('triton_per_fused__softmax_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_24(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 7
    RBLOCK: tl.constexpr = 8
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
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (896*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/f3/cf3qv5hme7xo4khxlui43opu2fb6prof7iqetvdgc3hebvgb5zr4.py
# Topologically Sorted Source Nodes: [k_softmax_10], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_10 => clone_82, exp_10, sub_38, sum_11
# Graph fragment:
#   %clone_82 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_117,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_82, %amax_10), kwargs = {})
#   %exp_10 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_38,), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_10, [2], True), kwargs = {})
triton_red_fused__softmax_25 = async_compile.triton('triton_red_fused__softmax_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_25(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7168
    rnumel = 113
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128) % 7
    x0 = xindex % 128
    x2 = (xindex // 896)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (113*x1)
        tmp1 = tl.full([1, 1], 785, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (128 + x0 + (384*r3) + (43392*x1) + (301440*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0 + (128*x2), [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pi/cpimwphzqgrkglugjrrs3d74rptgidfeuo2o7o2hcf47fz7zuxq5.py
# Topologically Sorted Source Nodes: [k_softmax_10], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_10 => clone_82, exp_10, sub_38, sum_11
# Graph fragment:
#   %clone_82 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_117,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_82, %amax_10), kwargs = {})
#   %exp_10 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_38,), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_10, [2], True), kwargs = {})
triton_per_fused__softmax_26 = async_compile.triton('triton_per_fused__softmax_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_26(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 7
    RBLOCK: tl.constexpr = 8
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
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (896*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pg/cpgjmzwqfmnbmjldtxjsabqkahijxtyvuzhnrbpg7rv6i3ykjw3d.py
# Topologically Sorted Source Nodes: [k_softmax_10], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_10 => clone_82, div_10, exp_10, sub_38
# Graph fragment:
#   %clone_82 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_117,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_82, %amax_10), kwargs = {})
#   %exp_10 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_38,), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_10, %sum_11), kwargs = {})
triton_poi_fused__softmax_27 = async_compile.triton('triton_poi_fused__softmax_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_27(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 785
    x2 = (xindex // 12560) % 8
    x3 = (xindex // 100480)
    x4 = (xindex // 12560)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (16*x2) + (384*x1) + (301440*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x4)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 + (16*x4)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tv/ctvmn6wwerhysy2ylkt6bfegxxvnaznher5wmyohhjvjlpavarun.py
# Topologically Sorted Source Nodes: [factor_att_20], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   factor_att_20 => clone_83
# Graph fragment:
#   %clone_83 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_47,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_28 = async_compile.triton('triton_poi_fused_clone_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 785
    x2 = (xindex // 12560) % 8
    x3 = (xindex // 100480)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (16*x2) + (384*x1) + (301440*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h5/ch5qrxauivatyfdisy7xbgqfkbkydn5tku65dmjppbvheqejogzf.py
# Topologically Sorted Source Nodes: [factor_att_21], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   factor_att_21 => clone_84
# Graph fragment:
#   %clone_84 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_48,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_29 = async_compile.triton('triton_poi_fused_clone_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 785
    x2 = (xindex // 12560) % 8
    x3 = (xindex // 100480)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (384*x1) + (301440*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zq/czqw5b4q3ni23om7st6rbyjmulsa6sx456kq5yv74aiweiat6ngz.py
# Topologically Sorted Source Nodes: [EV_hat_20, EV_hat_21, x_193], Original ATen: [aten.mul, aten.constant_pad_nd, aten.clone]
# Source node to ATen node mapping:
#   EV_hat_20 => mul_106
#   EV_hat_21 => constant_pad_nd_10
#   x_193 => clone_85
# Graph fragment:
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_143, %permute_128), kwargs = {})
#   %constant_pad_nd_10 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_106, [0, 0, 1, 0, 0, 0], 0.0), kwargs = {})
#   %clone_85 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_129,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_constant_pad_nd_mul_30 = async_compile.triton('triton_poi_fused_clone_constant_pad_nd_mul_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_constant_pad_nd_mul_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_constant_pad_nd_mul_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 128) % 785
    x4 = xindex % 128
    x5 = (xindex // 128)
    x0 = xindex % 16
    x1 = (xindex // 16) % 8
    x3 = (xindex // 100480)
    x6 = xindex
    tmp38 = tl.load(in_ptr7 + (x0 + (16*x2) + (12560*x1) + (100480*x3)), xmask)
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x4 + (384*x5)), tmp2 & xmask, other=0.0)
    tmp4 = x4
    tmp5 = tmp4 >= tmp1
    tmp6 = tl.full([1], 32, tl.int64)
    tmp7 = tmp4 < tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = tl.load(in_ptr1 + ((32*(((-1) + x2) % 784)) + (25088*x3) + (x0 + (16*x1))), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0 + (16*x1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tmp4 >= tmp6
    tmp15 = tl.full([1], 80, tl.int64)
    tmp16 = tmp4 < tmp15
    tmp17 = tmp14 & tmp16
    tmp18 = tmp17 & tmp2
    tmp19 = tl.load(in_ptr3 + ((48*(((-1) + x2) % 784)) + (37632*x3) + ((-32) + x0 + (16*x1))), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr4 + ((-32) + x0 + (16*x1)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp18, tmp21, tmp22)
    tmp24 = tmp4 >= tmp15
    tmp25 = tl.full([1], 128, tl.int64)
    tmp26 = tmp4 < tmp25
    tmp27 = tmp24 & tmp2
    tmp28 = tl.load(in_ptr5 + ((48*(((-1) + x2) % 784)) + (37632*x3) + ((-80) + x0 + (16*x1))), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-80) + x0 + (16*x1)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp27, tmp30, tmp31)
    tmp33 = tl.where(tmp17, tmp23, tmp32)
    tmp34 = tl.where(tmp7, tmp13, tmp33)
    tmp35 = tmp3 * tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp2, tmp35, tmp36)
    tmp39 = 0.25
    tmp40 = tmp38 * tmp39
    tmp41 = tmp40 + tmp37
    tl.store(out_ptr1 + (x6), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pn/cpnfckm2hzu72w2iphefhvz66kz6kz22du27vtt2kpkkhoibinxz.py
# Topologically Sorted Source Nodes: [x_190, x_196, x_197], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_190 => cat_26
#   x_196 => add_108
#   x_197 => add_109, add_110, mul_108, mul_109, rsqrt_28, sub_39, var_mean_28
# Graph fragment:
#   %cat_26 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_138, %permute_123], 1), kwargs = {})
#   %add_108 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_26, %view_226), kwargs = {})
#   %var_mean_28 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_108, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_108, %getitem_123), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_122, 1e-06), kwargs = {})
#   %rsqrt_28 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_109,), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %rsqrt_28), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_108, %arg57_1), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_109, %arg58_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_31 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 785
    r2 = rindex
    x1 = (xindex // 785)
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (r2 + (128*x3)), xmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (100480*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 785, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (128*(((-1) + x0) % 784)) + (100352*x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (128 + r2 + (128*(((-1) + x0) % 784)) + (100480*x1)), tmp6 & xmask, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp20 - tmp30
    tmp38 = 128.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp20, xmask)
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp47, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5y/c5yyb36s6qpmdckbbh5axnxwlsdydpreaagwulzer236ejhpidi3.py
# Topologically Sorted Source Nodes: [x_199], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_199 => add_111, erf_10, mul_110, mul_111, mul_112
# Graph fragment:
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_228, 0.5), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_228, 0.7071067811865476), kwargs = {})
#   %erf_10 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_111,), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_10, 1), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_110, %add_111), kwargs = {})
triton_poi_fused_gelu_32 = async_compile.triton('triton_poi_fused_gelu_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_32(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6430720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
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


# kernel path: /tmp/torchinductor_sahanp/zc/czcyy7pmc3vldovcuqdcnyqbbhx26qapwnvxvhisds7dvhcu2gah.py
# Topologically Sorted Source Nodes: [x_203], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_203 => add_112
# Graph fragment:
#   %add_112 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_108, %view_230), kwargs = {})
triton_poi_fused_add_33 = async_compile.triton('triton_poi_fused_add_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_33(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vs/cvsrody4wj2hzrpbmnz3ym3fsrvv4wqy2sbo6urutydqiy6iaczw.py
# Topologically Sorted Source Nodes: [x_206, x_207], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_206 => cat_28
#   x_207 => add_114, add_115, mul_113, mul_114, rsqrt_29, sub_40, var_mean_29
# Graph fragment:
#   %cat_28 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_150, %permute_134], 1), kwargs = {})
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_28, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_28, %getitem_125), kwargs = {})
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_124, 1e-06), kwargs = {})
#   %rsqrt_29 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_114,), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %rsqrt_29), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_113, %arg63_1), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_114, %arg64_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_34 = async_compile.triton('triton_per_fused_cat_native_layer_norm_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 785
    r2 = rindex
    x1 = (xindex // 785)
    x3 = xindex
    tmp40 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (128*x0) + (100480*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 785, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (128*(((-1) + x0) % 784)) + (100352*x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (128 + r2 + (128*(((-1) + x0) % 784)) + (100480*x1)), tmp6 & xmask, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp16 - tmp26
    tmp34 = 128.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp43, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/v2/cv2gz4ou46qfhlyfghrmhmnzn3h67mpjiibylx4ca7b5tlynclnw.py
# Topologically Sorted Source Nodes: [x_206, x_212, x_213], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_206 => cat_28
#   x_212 => add_117
#   x_213 => add_118, add_119, mul_117, mul_118, rsqrt_30, sub_42, var_mean_30
# Graph fragment:
#   %cat_28 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_150, %permute_134], 1), kwargs = {})
#   %add_117 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_28, %view_246), kwargs = {})
#   %var_mean_30 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_117, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_117, %getitem_133), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_132, 1e-06), kwargs = {})
#   %rsqrt_30 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_118,), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %rsqrt_30), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_117, %arg69_1), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_118, %arg70_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_35 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 785
    r2 = rindex
    x1 = (xindex // 785)
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (r2 + (128*x3)), xmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (128*x0) + (100480*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 785, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (128*(((-1) + x0) % 784)) + (100352*x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (128 + r2 + (128*(((-1) + x0) % 784)) + (100480*x1)), tmp6 & xmask, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp20 - tmp30
    tmp38 = 128.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp20, xmask)
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp47, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4u/c4ukenyihj4jf42goqd4y676rjg2tcufh6yrzc2vn4itigfnleup.py
# Topologically Sorted Source Nodes: [x2_nocls_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x2_nocls_1 => clone_96
# Graph fragment:
#   %clone_96 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_144,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_36 = async_compile.triton('triton_poi_fused_clone_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_36(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 100352)
    x3 = xindex % 100352
    x0 = xindex % 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x3 + (100480*x2)), None)
    tmp1 = tl.load(in_ptr1 + (128 + x3 + (100480*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bm/cbmtvuh6qukc34gmqbu3psgqu7tc5jcfu6bqkuk3vlwe5qolgx5k.py
# Topologically Sorted Source Nodes: [x2_nocls_1, x_220], Original ATen: [aten.clone, aten.convolution]
# Source node to ATen node mapping:
#   x2_nocls_1 => clone_96
#   x_220 => convolution_54
# Graph fragment:
#   %clone_96 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_144,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_54 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_96, %arg75_1, %arg76_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_clone_convolution_37 = async_compile.triton('triton_poi_fused_clone_convolution_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_37(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (512*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/y3/cy346iuiiwju2rva6jlzps26w7jmham2drkibne7yo6z3j5iebg6.py
# Topologically Sorted Source Nodes: [x_222], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_222 => add_122, add_123, clone_97, mul_122, mul_123, rsqrt_31, sub_43, var_mean_31
# Graph fragment:
#   %clone_97 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_145,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_31 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_97, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_97, %getitem_135), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_134, 1e-05), kwargs = {})
#   %rsqrt_31 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_122,), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %rsqrt_31), kwargs = {})
#   %mul_123 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_122, %arg77_1), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_123, %arg78_1), kwargs = {})
triton_per_fused_native_layer_norm_38 = async_compile.triton('triton_per_fused_native_layer_norm_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 196
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 320, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 320.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r1 + (320*x2) + (63040*x3)), tmp29, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mo/cmoex3hn3yezwsq7yvzaalr2o7lbrleqbhlzwpfbbzeahdulvk6y.py
# Topologically Sorted Source Nodes: [x_223], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_223 => cat_30
# Graph fragment:
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_54, %add_123], 1), kwargs = {})
triton_poi_fused_cat_39 = async_compile.triton('triton_poi_fused_cat_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 320
    x1 = (xindex // 320)
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0 + (63040*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/we/cwedbkcif3qc6ae3xamacwhekj2lpniu35ln2or4ocfiy7x7a5xc.py
# Topologically Sorted Source Nodes: [x_226, x_227], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_226 => cat_31
#   x_227 => add_125, add_126, mul_124, mul_125, rsqrt_32, sub_44, var_mean_32
# Graph fragment:
#   %cat_31 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_165, %permute_147], 1), kwargs = {})
#   %var_mean_32 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_31, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_31, %getitem_137), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_136, 1e-06), kwargs = {})
#   %rsqrt_32 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_125,), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %rsqrt_32), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %arg82_1), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %arg83_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_40 = async_compile.triton('triton_per_fused_cat_native_layer_norm_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp40 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (63040*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (320*(((-1) + x0) % 196)) + (62720*x1)), rmask & tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (320 + r2 + (320*(((-1) + x0) % 196)) + (63040*x1)), rmask & tmp6, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 320, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 320.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr3 + (r2 + (320*x3)), tmp43, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ht/chtdzalxpu7dt3spiluaau245wvtr7wkqw3qrohegbwwjpjvdgez.py
# Topologically Sorted Source Nodes: [k_softmax_12], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_12 => amax_12, clone_98
# Graph fragment:
#   %clone_98 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_139,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_12 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_98, [2], True), kwargs = {})
triton_red_fused__softmax_41 = async_compile.triton('triton_red_fused__softmax_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_41(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 99
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320) % 2
    x0 = xindex % 320
    x2 = (xindex // 640)
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (99*x1)
        tmp1 = tl.full([1, 1], 197, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (320 + x0 + (960*r3) + (95040*x1) + (189120*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sm/csmvun7ljey7u4if4ttjzjtqdkwpiwttevugy57ypxwonwtla4uc.py
# Topologically Sorted Source Nodes: [k_softmax_12], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_12 => amax_12, clone_98
# Graph fragment:
#   %clone_98 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_139,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_12 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_98, [2], True), kwargs = {})
triton_per_fused__softmax_42 = async_compile.triton('triton_per_fused__softmax_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_42(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 320
    x1 = (xindex // 320)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (640*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/iy/ciy76rr4zkzgvtpljalluslp5wugiwd4wzsaj3qvb63tkysxvo56.py
# Topologically Sorted Source Nodes: [k_softmax_12], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_12 => clone_98, exp_12, sub_45, sum_13
# Graph fragment:
#   %clone_98 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_139,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_98, %amax_12), kwargs = {})
#   %exp_12 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_45,), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_12, [2], True), kwargs = {})
triton_red_fused__softmax_43 = async_compile.triton('triton_red_fused__softmax_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_43(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 99
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320) % 2
    x0 = xindex % 320
    x2 = (xindex // 640)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (99*x1)
        tmp1 = tl.full([1, 1], 197, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (320 + x0 + (960*r3) + (95040*x1) + (189120*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0 + (320*x2), [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4u/c4uj3ex6ddl54nznlrsqgcbecepiglcdzoe74jvusjtkv4sgwstc.py
# Topologically Sorted Source Nodes: [k_softmax_12], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_12 => clone_98, exp_12, sub_45, sum_13
# Graph fragment:
#   %clone_98 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_139,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_98, %amax_12), kwargs = {})
#   %exp_12 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_45,), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_12, [2], True), kwargs = {})
triton_per_fused__softmax_44 = async_compile.triton('triton_per_fused__softmax_44', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_44(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 320
    x1 = (xindex // 320)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (640*x1)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bc/cbc3utzsoevts446u576nxusyinlfxnhkhvaqdxfmwglevqro343.py
# Topologically Sorted Source Nodes: [k_softmax_12], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_12 => clone_98, div_12, exp_12, sub_45
# Graph fragment:
#   %clone_98 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_139,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_98, %amax_12), kwargs = {})
#   %exp_12 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_45,), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_12, %sum_13), kwargs = {})
triton_poi_fused__softmax_45 = async_compile.triton('triton_poi_fused__softmax_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_45(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40) % 197
    x2 = (xindex // 7880) % 8
    x3 = (xindex // 63040)
    x4 = (xindex // 7880)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (320 + x0 + (40*x2) + (960*x1) + (189120*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (40*x4)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 + (40*x4)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hp/chpei7axff3eb6rzt2dk5zjyv5u6ky23gdpes72qscib3pimqey5.py
# Topologically Sorted Source Nodes: [factor_att_24], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   factor_att_24 => clone_99
# Graph fragment:
#   %clone_99 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_56,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_46 = async_compile.triton('triton_poi_fused_clone_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_46(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40) % 197
    x2 = (xindex // 7880) % 8
    x3 = (xindex // 63040)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (640 + x0 + (40*x2) + (960*x1) + (189120*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/z6/cz62cm6tb24hf7653ardou5hu5cybzonm4hdzd6zorybihl4jc47.py
# Topologically Sorted Source Nodes: [factor_att_25], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   factor_att_25 => clone_100
# Graph fragment:
#   %clone_100 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_57,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_47 = async_compile.triton('triton_poi_fused_clone_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40) % 197
    x2 = (xindex // 7880) % 8
    x3 = (xindex // 63040)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (40*x2) + (960*x1) + (189120*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zo/czosoetnlr5nntrzeqxlhrlnq56ouncjloatn4kknwf3epbi27j7.py
# Topologically Sorted Source Nodes: [EV_hat_24, EV_hat_25, x_229], Original ATen: [aten.mul, aten.constant_pad_nd, aten.clone]
# Source node to ATen node mapping:
#   EV_hat_24 => mul_126
#   EV_hat_25 => constant_pad_nd_12
#   x_229 => clone_101
# Graph fragment:
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_170, %permute_152), kwargs = {})
#   %constant_pad_nd_12 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_126, [0, 0, 1, 0, 0, 0], 0.0), kwargs = {})
#   %clone_101 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_153,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_constant_pad_nd_mul_48 = async_compile.triton('triton_poi_fused_clone_constant_pad_nd_mul_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_constant_pad_nd_mul_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_constant_pad_nd_mul_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 320) % 197
    x4 = xindex % 320
    x5 = (xindex // 320)
    x0 = xindex % 40
    x1 = (xindex // 40) % 8
    x3 = (xindex // 63040)
    x6 = xindex
    tmp38 = tl.load(in_ptr7 + (x0 + (40*x2) + (7880*x1) + (63040*x3)), xmask)
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x4 + (960*x5)), tmp2 & xmask, other=0.0)
    tmp4 = x4
    tmp5 = tmp4 >= tmp1
    tmp6 = tl.full([1], 80, tl.int64)
    tmp7 = tmp4 < tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = tl.load(in_ptr1 + ((80*(((-1) + x2) % 196)) + (15680*x3) + (x0 + (40*x1))), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0 + (40*x1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tmp4 >= tmp6
    tmp15 = tl.full([1], 200, tl.int64)
    tmp16 = tmp4 < tmp15
    tmp17 = tmp14 & tmp16
    tmp18 = tmp17 & tmp2
    tmp19 = tl.load(in_ptr3 + ((120*(((-1) + x2) % 196)) + (23520*x3) + ((-80) + x0 + (40*x1))), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr4 + ((-80) + x0 + (40*x1)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp18, tmp21, tmp22)
    tmp24 = tmp4 >= tmp15
    tmp25 = tl.full([1], 320, tl.int64)
    tmp26 = tmp4 < tmp25
    tmp27 = tmp24 & tmp2
    tmp28 = tl.load(in_ptr5 + ((120*(((-1) + x2) % 196)) + (23520*x3) + ((-200) + x0 + (40*x1))), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-200) + x0 + (40*x1)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp27, tmp30, tmp31)
    tmp33 = tl.where(tmp17, tmp23, tmp32)
    tmp34 = tl.where(tmp7, tmp13, tmp33)
    tmp35 = tmp3 * tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp2, tmp35, tmp36)
    tmp39 = 0.15811388300841897
    tmp40 = tmp38 * tmp39
    tmp41 = tmp40 + tmp37
    tl.store(out_ptr1 + (x6), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n3/cn3dkxl7p32sr6awkj7hzslhkcqdba6vjeoiup26hvgr4rcknxby.py
# Topologically Sorted Source Nodes: [x_226, x_232, x_233], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_226 => cat_31
#   x_232 => add_128
#   x_233 => add_129, add_130, mul_128, mul_129, rsqrt_33, sub_46, var_mean_33
# Graph fragment:
#   %cat_31 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_165, %permute_147], 1), kwargs = {})
#   %add_128 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_31, %view_268), kwargs = {})
#   %var_mean_33 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_128, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_128, %getitem_145), kwargs = {})
#   %add_129 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_144, 1e-06), kwargs = {})
#   %rsqrt_33 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_129,), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %rsqrt_33), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_128, %arg94_1), kwargs = {})
#   %add_130 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_129, %arg95_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_49 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_49', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (r2 + (320*x3)), rmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (63040*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (320*(((-1) + x0) % 196)) + (62720*x1)), rmask & tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (320 + r2 + (320*(((-1) + x0) % 196)) + (63040*x1)), rmask & tmp6, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tl.full([1], 320, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = tmp20 - tmp30
    tmp38 = 320.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(in_out_ptr0 + (r2 + (320*x3)), tmp20, rmask)
    tl.store(out_ptr2 + (r2 + (320*x3)), tmp47, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jx/cjxez3vuk77gqs63nnwpup6omlr7v4wfmvvq2xiqnfr3c47zfn7o.py
# Topologically Sorted Source Nodes: [x_235], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_235 => add_131, erf_12, mul_130, mul_131, mul_132
# Graph fragment:
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_270, 0.5), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_270, 0.7071067811865476), kwargs = {})
#   %erf_12 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_131,), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_12, 1), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %add_131), kwargs = {})
triton_poi_fused_gelu_50 = async_compile.triton('triton_poi_fused_gelu_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_50(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2017280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/aj/cajlaltwzcyvp5tvx5u3otpafzjdr6lblqkplrmkftf326erx55l.py
# Topologically Sorted Source Nodes: [x_239], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_239 => add_132
# Graph fragment:
#   %add_132 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_128, %view_272), kwargs = {})
triton_poi_fused_add_51 = async_compile.triton('triton_poi_fused_add_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_51(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zz/czzgzep66kogrwj5plgyqsg62q53rcqc7bsabcvh7ztykiwjvf73.py
# Topologically Sorted Source Nodes: [x_242, x_243], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_242 => cat_33
#   x_243 => add_134, add_135, mul_133, mul_134, rsqrt_34, sub_47, var_mean_34
# Graph fragment:
#   %cat_33 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_177, %permute_158], 1), kwargs = {})
#   %var_mean_34 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_33, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_33, %getitem_147), kwargs = {})
#   %add_134 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_146, 1e-06), kwargs = {})
#   %rsqrt_34 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_134,), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %rsqrt_34), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %arg100_1), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, %arg101_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_52 = async_compile.triton('triton_per_fused_cat_native_layer_norm_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp40 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (320*x0) + (63040*x1)), rmask & tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (320*(((-1) + x0) % 196)) + (62720*x1)), rmask & tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (320 + r2 + (320*(((-1) + x0) % 196)) + (63040*x1)), rmask & tmp6, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 320, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 320.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr3 + (r2 + (320*x3)), tmp43, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vh/cvhz6upwb5kghvc5ibujmymgrjeeupeakl7tbsdldb4nf4jks7fk.py
# Topologically Sorted Source Nodes: [x_242, x_248, x_249], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_242 => cat_33
#   x_248 => add_137
#   x_249 => add_138, add_139, mul_137, mul_138, rsqrt_35, sub_49, var_mean_35
# Graph fragment:
#   %cat_33 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_177, %permute_158], 1), kwargs = {})
#   %add_137 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_33, %view_288), kwargs = {})
#   %var_mean_35 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_137, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_137, %getitem_155), kwargs = {})
#   %add_138 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_154, 1e-06), kwargs = {})
#   %rsqrt_35 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_138,), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %rsqrt_35), kwargs = {})
#   %mul_138 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_137, %arg106_1), kwargs = {})
#   %add_139 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_138, %arg107_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_53 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_53', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_53(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (r2 + (320*x3)), rmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (320*x0) + (63040*x1)), rmask & tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (320*(((-1) + x0) % 196)) + (62720*x1)), rmask & tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (320 + r2 + (320*(((-1) + x0) % 196)) + (63040*x1)), rmask & tmp6, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tl.full([1], 320, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = tmp20 - tmp30
    tmp38 = 320.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(in_out_ptr0 + (r2 + (320*x3)), tmp20, rmask)
    tl.store(out_ptr2 + (r2 + (320*x3)), tmp47, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/t3/ct35cwglv7oguu2ubxlc6bvxttdzuvw5qg43ns2ovxa6fejekycz.py
# Topologically Sorted Source Nodes: [x3_nocls_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x3_nocls_1 => clone_112
# Graph fragment:
#   %clone_112 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_168,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_54 = async_compile.triton('triton_poi_fused_clone_54', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_54(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 62720)
    x3 = xindex % 62720
    x0 = xindex % 320
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (320 + x3 + (63040*x2)), xmask)
    tmp1 = tl.load(in_ptr1 + (320 + x3 + (63040*x2)), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g4/cg4xkl3cur2zr4iyh5gc2mwo3qsiqebu4grcjo2ppen3xqxzned3.py
# Topologically Sorted Source Nodes: [x3_nocls_1, x_256], Original ATen: [aten.clone, aten.convolution]
# Source node to ATen node mapping:
#   x3_nocls_1 => clone_112
#   x_256 => convolution_63
# Graph fragment:
#   %clone_112 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_168,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_63 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_112, %arg112_1, %arg113_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_clone_convolution_55 = async_compile.triton('triton_poi_fused_clone_convolution_55', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[262144, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_convolution_55(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 163840
    xnumel = 4
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (1280*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ya/cyaala64hei5oe4saqvoecq6ccrw67w6cxtwbn3pqy6m5cv3gwkj.py
# Topologically Sorted Source Nodes: [x_258], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_258 => add_142, add_143, clone_113, mul_142, mul_143, rsqrt_36, sub_50, var_mean_36
# Graph fragment:
#   %clone_113 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_169,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_36 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_113, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_113, %getitem_157), kwargs = {})
#   %add_142 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_156, 1e-05), kwargs = {})
#   %rsqrt_36 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_142,), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %rsqrt_36), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_142, %arg114_1), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_143, %arg115_1), kwargs = {})
triton_per_fused_native_layer_norm_56 = async_compile.triton('triton_per_fused_native_layer_norm_56', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 49
    x3 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp2 - tmp10
    tmp17 = 512.0
    tmp18 = tmp15 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tl.store(out_ptr2 + (r1 + (512*x2) + (25600*x3)), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ql/cqlhf2c5a6s73oik25al3u6bdc45mg4an7rehaoj7zkprzizfbln.py
# Topologically Sorted Source Nodes: [x_259], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_259 => cat_35
# Graph fragment:
#   %cat_35 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_63, %add_143], 1), kwargs = {})
triton_poi_fused_cat_57 = async_compile.triton('triton_poi_fused_cat_57', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_57(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0 + (25600*x1)), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nu/cnu63xhelxnfzq43akpm64puvbdpz6fyoz2krayhat464lgw4t2e.py
# Topologically Sorted Source Nodes: [x_262, x_263], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_262 => cat_36
#   x_263 => add_145, add_146, mul_144, mul_145, rsqrt_37, sub_51, var_mean_37
# Graph fragment:
#   %cat_36 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_192, %permute_171], 1), kwargs = {})
#   %var_mean_37 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_36, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_36, %getitem_159), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_158, 1e-06), kwargs = {})
#   %rsqrt_37 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_145,), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %rsqrt_37), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_144, %arg119_1), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_145, %arg120_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_58 = async_compile.triton('triton_per_fused_cat_native_layer_norm_58', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex % 50
    r2 = rindex
    x1 = (xindex // 50)
    x3 = xindex
    tmp37 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (25600*x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 50, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (512*(((-1) + x0) % 49)) + (25088*x1)), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (512 + r2 + (512*(((-1) + x0) % 49)) + (25600*x1)), tmp6, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tl.full([1], 512, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp17 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp30 = tmp16 - tmp24
    tmp31 = 512.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp40, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jw/cjwwf3hykilvqpyzf6jzllasioirj777yxbgwdyzr7edl4wlxdat.py
# Topologically Sorted Source Nodes: [k_softmax_14], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_14 => amax_14, clone_114, exp_14, sub_52, sum_15
# Graph fragment:
#   %clone_114 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_161,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_14 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_114, [2], True), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_114, %amax_14), kwargs = {})
#   %exp_14 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_52,), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_14, [2], True), kwargs = {})
triton_per_fused__softmax_59 = async_compile.triton('triton_per_fused__softmax_59', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_59(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 50
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (1536*r2) + (76800*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ho/choj4pv4gmaj5pauhevork6u64a4p2kxr4wzoupiprggnfain5u4.py
# Topologically Sorted Source Nodes: [k_softmax_14], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   k_softmax_14 => clone_114, div_14, exp_14, sub_52
# Graph fragment:
#   %clone_114 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_161,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_114, %amax_14), kwargs = {})
#   %exp_14 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_52,), kwargs = {})
#   %div_14 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_14, %sum_15), kwargs = {})
triton_poi_fused__softmax_60 = async_compile.triton('triton_poi_fused__softmax_60', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_60(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 50
    x2 = (xindex // 3200) % 8
    x3 = (xindex // 25600)
    x4 = (xindex // 3200)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (64*x2) + (1536*x1) + (76800*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x4)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 + (64*x4)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ao/caou6vb74quwhyieklwb36ua4z4er7bf6qzhyvsqntk66add34hm.py
# Topologically Sorted Source Nodes: [factor_att_28], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   factor_att_28 => clone_115
# Graph fragment:
#   %clone_115 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_65,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_61 = async_compile.triton('triton_poi_fused_clone_61', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_61(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 50
    x2 = (xindex // 3200) % 8
    x3 = (xindex // 25600)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (64*x2) + (1536*x1) + (76800*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vh/cvhocbtwriozsvwhnfk25zyweovgdypybn4tbwyfyihvdclmi5a4.py
# Topologically Sorted Source Nodes: [factor_att_29], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   factor_att_29 => clone_116
# Graph fragment:
#   %clone_116 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_66,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_62 = async_compile.triton('triton_poi_fused_clone_62', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_62(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 50
    x2 = (xindex // 3200) % 8
    x3 = (xindex // 25600)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1536*x1) + (76800*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7a/c7a42qpav6p6gtjt36vsym6psilsmymgsqscnmeb5b6uw67odpvx.py
# Topologically Sorted Source Nodes: [EV_hat_28, EV_hat_29, x_265], Original ATen: [aten.mul, aten.constant_pad_nd, aten.clone]
# Source node to ATen node mapping:
#   EV_hat_28 => mul_146
#   EV_hat_29 => constant_pad_nd_14
#   x_265 => clone_117
# Graph fragment:
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_197, %permute_176), kwargs = {})
#   %constant_pad_nd_14 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_146, [0, 0, 1, 0, 0, 0], 0.0), kwargs = {})
#   %clone_117 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_177,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_constant_pad_nd_mul_63 = async_compile.triton('triton_poi_fused_clone_constant_pad_nd_mul_63', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_constant_pad_nd_mul_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_constant_pad_nd_mul_63(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 512) % 50
    x4 = xindex % 512
    x5 = (xindex // 512)
    x0 = xindex % 64
    x1 = (xindex // 64) % 8
    x3 = (xindex // 25600)
    x6 = xindex
    tmp38 = tl.load(in_ptr7 + (x0 + (64*x2) + (3200*x1) + (25600*x3)), None)
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x4 + (1536*x5)), tmp2, other=0.0)
    tmp4 = x4
    tmp5 = tmp4 >= tmp1
    tmp6 = tl.full([1], 128, tl.int64)
    tmp7 = tmp4 < tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = tl.load(in_ptr1 + ((128*(((-1) + x2) % 49)) + (6272*x3) + (x0 + (64*x1))), tmp8, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0 + (64*x1)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tmp4 >= tmp6
    tmp15 = tl.full([1], 320, tl.int64)
    tmp16 = tmp4 < tmp15
    tmp17 = tmp14 & tmp16
    tmp18 = tmp17 & tmp2
    tmp19 = tl.load(in_ptr3 + ((192*(((-1) + x2) % 49)) + (9408*x3) + ((-128) + x0 + (64*x1))), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr4 + ((-128) + x0 + (64*x1)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp18, tmp21, tmp22)
    tmp24 = tmp4 >= tmp15
    tmp25 = tl.full([1], 512, tl.int64)
    tmp26 = tmp4 < tmp25
    tmp27 = tmp24 & tmp2
    tmp28 = tl.load(in_ptr5 + ((192*(((-1) + x2) % 49)) + (9408*x3) + ((-320) + x0 + (64*x1))), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-320) + x0 + (64*x1)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp27, tmp30, tmp31)
    tmp33 = tl.where(tmp17, tmp23, tmp32)
    tmp34 = tl.where(tmp7, tmp13, tmp33)
    tmp35 = tmp3 * tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp2, tmp35, tmp36)
    tmp39 = 0.125
    tmp40 = tmp38 * tmp39
    tmp41 = tmp40 + tmp37
    tl.store(out_ptr1 + (x6), tmp41, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yk/cykil367zywigpbfvn22iuzagfsa3bmloto2bqg6m6fljt45cgcv.py
# Topologically Sorted Source Nodes: [x_262, x_268, x_269], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_262 => cat_36
#   x_268 => add_148
#   x_269 => add_149, add_150, mul_148, mul_149, rsqrt_38, sub_53, var_mean_38
# Graph fragment:
#   %cat_36 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_192, %permute_171], 1), kwargs = {})
#   %add_148 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_36, %view_310), kwargs = {})
#   %var_mean_38 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_148, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_148, %getitem_167), kwargs = {})
#   %add_149 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_166, 1e-06), kwargs = {})
#   %rsqrt_38 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_149,), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %rsqrt_38), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %arg131_1), kwargs = {})
#   %add_150 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %arg132_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_64 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_64', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_64', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_64(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex % 50
    r2 = rindex
    x1 = (xindex // 50)
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (r2 + (512*x3)), None)
    tmp18 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (25600*x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 50, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (512*(((-1) + x0) % 49)) + (25088*x1)), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (512 + r2 + (512*(((-1) + x0) % 49)) + (25600*x1)), tmp6, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = tl.full([1], 512, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp21 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp34 = tmp20 - tmp28
    tmp35 = 512.0
    tmp36 = tmp33 / tmp35
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp20, None)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp44, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ap/cap6wqvcnrl7r54vj2urui25usovw5qimekvnsohqkbqegqjpfvc.py
# Topologically Sorted Source Nodes: [x_271], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_271 => add_151, erf_14, mul_150, mul_151, mul_152
# Graph fragment:
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_312, 0.5), kwargs = {})
#   %mul_151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_312, 0.7071067811865476), kwargs = {})
#   %erf_14 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_151,), kwargs = {})
#   %add_151 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_14, 1), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_150, %add_151), kwargs = {})
triton_poi_fused_gelu_65 = async_compile.triton('triton_poi_fused_gelu_65', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_65(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 2048
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


# kernel path: /tmp/torchinductor_sahanp/zj/czj2dxietmz23bwu24jkq4lvl27tpuawufaacwuznuesyqkpeonr.py
# Topologically Sorted Source Nodes: [x_275], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_275 => add_152
# Graph fragment:
#   %add_152 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_148, %view_314), kwargs = {})
triton_poi_fused_add_66 = async_compile.triton('triton_poi_fused_add_66', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_66', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_66(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wj/cwjtonkq6gbuyer5qab2nllwd5pbroc24dzwz7jcdwlcdi2wbl2n.py
# Topologically Sorted Source Nodes: [x_278, x_279], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_278 => cat_38
#   x_279 => add_154, add_155, mul_153, mul_154, rsqrt_39, sub_54, var_mean_39
# Graph fragment:
#   %cat_38 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_204, %permute_182], 1), kwargs = {})
#   %var_mean_39 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_38, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_38, %getitem_169), kwargs = {})
#   %add_154 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_168, 1e-06), kwargs = {})
#   %rsqrt_39 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_154,), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %rsqrt_39), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_153, %arg137_1), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_154, %arg138_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_67 = async_compile.triton('triton_per_fused_cat_native_layer_norm_67', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex % 50
    r2 = rindex
    x1 = (xindex // 50)
    x3 = xindex
    tmp37 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (512*x0) + (25600*x1)), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 50, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (512*(((-1) + x0) % 49)) + (25088*x1)), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (512 + r2 + (512*(((-1) + x0) % 49)) + (25600*x1)), tmp6, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tl.full([1], 512, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp17 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp30 = tmp16 - tmp24
    tmp31 = 512.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp40, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d2/cd242npxv62dxckhd3vtev62gwi5mrcvrt5rd6bt6vmyefcmboyu.py
# Topologically Sorted Source Nodes: [x_278, x_284, x_285], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_278 => cat_38
#   x_284 => add_157
#   x_285 => add_158, add_159, mul_157, mul_158, rsqrt_40, sub_56, var_mean_40
# Graph fragment:
#   %cat_38 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_204, %permute_182], 1), kwargs = {})
#   %add_157 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_38, %view_330), kwargs = {})
#   %var_mean_40 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_157, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_157, %getitem_177), kwargs = {})
#   %add_158 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_176, 1e-06), kwargs = {})
#   %rsqrt_40 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_158,), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %rsqrt_40), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_157, %arg143_1), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_158, %arg144_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_68 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_68', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_68', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_68(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex % 50
    r2 = rindex
    x1 = (xindex // 50)
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (r2 + (512*x3)), None)
    tmp18 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (512*x0) + (25600*x1)), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 50, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (512*(((-1) + x0) % 49)) + (25088*x1)), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (512 + r2 + (512*(((-1) + x0) % 49)) + (25600*x1)), tmp6, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = tl.full([1], 512, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp21 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp34 = tmp20 - tmp28
    tmp35 = 512.0
    tmp36 = tmp33 / tmp35
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp20, None)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp44, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eb/cebz6ezsjrjwpkzrezmr32hxsrewf4gxfgiz7gnxll4tku2mopko.py
# Topologically Sorted Source Nodes: [x_291, x_292], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_291 => add_161
#   x_292 => var_mean_41
# Graph fragment:
#   %add_161 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_157, %view_334), kwargs = {})
#   %var_mean_41 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_161, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_69 = async_compile.triton('triton_per_fused_add_native_layer_norm_69', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_69(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 512, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tl.store(out_ptr0 + (x0), tmp12, None)
    tl.store(out_ptr1 + (x0), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3j/c3jog6jejgtjnloja5bvfwdbblqigvhdeoesro5izh3d6mkgne5h.py
# Topologically Sorted Source Nodes: [x_294], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_294 => clone_129
# Graph fragment:
#   %clone_129 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%select_1,), kwargs = {})
triton_poi_fused_clone_70 = async_compile.triton('triton_poi_fused_clone_70', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_70(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (25600*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (25600*x1)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (50*x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (50*x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 512.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (1, 1, 64), (64, 64, 1))
    assert_size_stride(arg6_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (192, 64), (64, 1))
    assert_size_stride(arg11_1, (192, ), (1, ))
    assert_size_stride(arg12_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg13_1, (16, ), (1, ))
    assert_size_stride(arg14_1, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg15_1, (24, ), (1, ))
    assert_size_stride(arg16_1, (24, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (64, 64), (64, 1))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (64, ), (1, ))
    assert_size_stride(arg22_1, (512, 64), (64, 1))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (64, 512), (512, 1))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, ), (1, ))
    assert_size_stride(arg27_1, (64, ), (1, ))
    assert_size_stride(arg28_1, (192, 64), (64, 1))
    assert_size_stride(arg29_1, (192, ), (1, ))
    assert_size_stride(arg30_1, (64, 64), (64, 1))
    assert_size_stride(arg31_1, (64, ), (1, ))
    assert_size_stride(arg32_1, (64, ), (1, ))
    assert_size_stride(arg33_1, (64, ), (1, ))
    assert_size_stride(arg34_1, (512, 64), (64, 1))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (64, 512), (512, 1))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (128, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (1, 1, 128), (128, 128, 1))
    assert_size_stride(arg43_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (384, 128), (128, 1))
    assert_size_stride(arg48_1, (384, ), (1, ))
    assert_size_stride(arg49_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg50_1, (32, ), (1, ))
    assert_size_stride(arg51_1, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg52_1, (48, ), (1, ))
    assert_size_stride(arg53_1, (48, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg54_1, (48, ), (1, ))
    assert_size_stride(arg55_1, (128, 128), (128, 1))
    assert_size_stride(arg56_1, (128, ), (1, ))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (1024, 128), (128, 1))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (128, 1024), (1024, 1))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (384, 128), (128, 1))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (128, 128), (128, 1))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (1024, 128), (128, 1))
    assert_size_stride(arg72_1, (1024, ), (1, ))
    assert_size_stride(arg73_1, (128, 1024), (1024, 1))
    assert_size_stride(arg74_1, (128, ), (1, ))
    assert_size_stride(arg75_1, (320, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(arg76_1, (320, ), (1, ))
    assert_size_stride(arg77_1, (320, ), (1, ))
    assert_size_stride(arg78_1, (320, ), (1, ))
    assert_size_stride(arg79_1, (1, 1, 320), (320, 320, 1))
    assert_size_stride(arg80_1, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg81_1, (320, ), (1, ))
    assert_size_stride(arg82_1, (320, ), (1, ))
    assert_size_stride(arg83_1, (320, ), (1, ))
    assert_size_stride(arg84_1, (960, 320), (320, 1))
    assert_size_stride(arg85_1, (960, ), (1, ))
    assert_size_stride(arg86_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg87_1, (80, ), (1, ))
    assert_size_stride(arg88_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg89_1, (120, ), (1, ))
    assert_size_stride(arg90_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg91_1, (120, ), (1, ))
    assert_size_stride(arg92_1, (320, 320), (320, 1))
    assert_size_stride(arg93_1, (320, ), (1, ))
    assert_size_stride(arg94_1, (320, ), (1, ))
    assert_size_stride(arg95_1, (320, ), (1, ))
    assert_size_stride(arg96_1, (1280, 320), (320, 1))
    assert_size_stride(arg97_1, (1280, ), (1, ))
    assert_size_stride(arg98_1, (320, 1280), (1280, 1))
    assert_size_stride(arg99_1, (320, ), (1, ))
    assert_size_stride(arg100_1, (320, ), (1, ))
    assert_size_stride(arg101_1, (320, ), (1, ))
    assert_size_stride(arg102_1, (960, 320), (320, 1))
    assert_size_stride(arg103_1, (960, ), (1, ))
    assert_size_stride(arg104_1, (320, 320), (320, 1))
    assert_size_stride(arg105_1, (320, ), (1, ))
    assert_size_stride(arg106_1, (320, ), (1, ))
    assert_size_stride(arg107_1, (320, ), (1, ))
    assert_size_stride(arg108_1, (1280, 320), (320, 1))
    assert_size_stride(arg109_1, (1280, ), (1, ))
    assert_size_stride(arg110_1, (320, 1280), (1280, 1))
    assert_size_stride(arg111_1, (320, ), (1, ))
    assert_size_stride(arg112_1, (512, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (1, 1, 512), (512, 512, 1))
    assert_size_stride(arg117_1, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (1536, 512), (512, 1))
    assert_size_stride(arg122_1, (1536, ), (1, ))
    assert_size_stride(arg123_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg124_1, (128, ), (1, ))
    assert_size_stride(arg125_1, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg126_1, (192, ), (1, ))
    assert_size_stride(arg127_1, (192, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg128_1, (192, ), (1, ))
    assert_size_stride(arg129_1, (512, 512), (512, 1))
    assert_size_stride(arg130_1, (512, ), (1, ))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (512, ), (1, ))
    assert_size_stride(arg133_1, (2048, 512), (512, 1))
    assert_size_stride(arg134_1, (2048, ), (1, ))
    assert_size_stride(arg135_1, (512, 2048), (2048, 1))
    assert_size_stride(arg136_1, (512, ), (1, ))
    assert_size_stride(arg137_1, (512, ), (1, ))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (1536, 512), (512, 1))
    assert_size_stride(arg140_1, (1536, ), (1, ))
    assert_size_stride(arg141_1, (512, 512), (512, 1))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (512, ), (1, ))
    assert_size_stride(arg145_1, (2048, 512), (512, 1))
    assert_size_stride(arg146_1, (2048, ), (1, ))
    assert_size_stride(arg147_1, (512, 2048), (2048, 1))
    assert_size_stride(arg148_1, (512, ), (1, ))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (512, ), (1, ))
    assert_size_stride(arg151_1, (1000, 512), (512, 1))
    assert_size_stride(arg152_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_148], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg0_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((64, 3, 4, 4), (48, 1, 12, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_148], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg1_1, buf1, 192, 16, grid=grid(192, 16), stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [x_148], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del buf0
        del buf1
        buf8 = empty_strided_cuda((8, 3137, 64), (200768, 64, 1), torch.float32)
        buf7 = reinterpret_tensor(buf8, (8, 3136, 64), (200768, 64, 1), 64)  # alias
        # Topologically Sorted Source Nodes: [x_150], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2.run(buf2, arg2_1, arg3_1, arg4_1, buf7, 25088, 64, grid=grid(25088), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del buf2
        buf6 = reinterpret_tensor(buf8, (8, 1, 64), (200768, 64, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_151], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg5_1, buf6, 512, grid=grid(512), stream=stream0)
        del arg5_1
        del buf6
        del buf7
        # Topologically Sorted Source Nodes: [conv2d_37], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(reinterpret_tensor(buf8, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf9, (8, 64, 56, 56), (200704, 1, 3584, 64))
        buf14 = empty_strided_cuda((8, 3137, 64), (200768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_154, x_155], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_4.run(buf8, buf9, arg7_1, arg8_1, arg9_1, buf14, 25096, 64, grid=grid(25096), stream=stream0)
        del arg8_1
        del arg9_1
        buf15 = empty_strided_cuda((25096, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf14, (25096, 64), (64, 1), 0), reinterpret_tensor(arg10_1, (64, 192), (1, 64), 0), alpha=1, beta=1, out=buf15)
        del arg10_1
        del arg11_1
        buf16 = empty_strided_cuda((8, 8, 1, 8, 25), (1600, 8, 12800, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [k_softmax_8], Original ATen: [aten._softmax]
        triton_red_fused__softmax_5.run(buf15, buf16, 12800, 126, grid=grid(12800), stream=stream0)
        buf17 = empty_strided_cuda((8, 8, 1, 8), (64, 8, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_softmax_8], Original ATen: [aten._softmax]
        triton_per_fused__softmax_6.run(buf16, buf17, 512, 25, grid=grid(512), stream=stream0)
        buf18 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_8], Original ATen: [aten._softmax]
        triton_red_fused__softmax_7.run(buf15, buf17, buf18, 12800, 126, grid=grid(12800), stream=stream0)
        buf19 = empty_strided_cuda((8, 8, 1, 8), (64, 8, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_softmax_8], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf18, buf19, 512, 25, grid=grid(512), stream=stream0)
        buf20 = reinterpret_tensor(buf14, (8, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_8], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_9.run(buf15, buf17, buf19, buf20, 1606144, grid=grid(1606144), stream=stream0)
        buf21 = empty_strided_cuda((8, 8, 3137, 8), (200768, 25096, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [factor_att_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf15, buf21, 1606144, grid=grid(1606144), stream=stream0)
        buf22 = empty_strided_cuda((64, 8, 8), (64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [factor_att_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf21, (64, 3137, 8), (25096, 8, 1), 0), out=buf22)
        buf23 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [factor_att_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf15, buf23, 1606144, grid=grid(1606144), stream=stream0)
        buf24 = reinterpret_tensor(buf20, (64, 3137, 8), (25096, 8, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [factor_att_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf23, (64, 3137, 8), (25096, 8, 1), 0), buf22, out=buf24)
        # Topologically Sorted Source Nodes: [conv2d_38], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(reinterpret_tensor(buf15, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), arg12_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf25, (8, 16, 56, 56), (50176, 1, 896, 16))
        # Topologically Sorted Source Nodes: [conv2d_39], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(reinterpret_tensor(buf15, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), arg14_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf26, (8, 24, 56, 56), (75264, 1, 1344, 24))
        # Topologically Sorted Source Nodes: [conv2d_40], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(reinterpret_tensor(buf15, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), arg16_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf27, (8, 24, 56, 56), (75264, 1, 1344, 24))
        buf29 = reinterpret_tensor(buf23, (8, 3137, 8, 8), (200768, 64, 8, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [EV_hat_16, EV_hat_17, x_157], Original ATen: [aten.mul, aten.constant_pad_nd, aten.clone]
        triton_poi_fused_clone_constant_pad_nd_mul_12.run(buf15, buf25, arg13_1, buf26, arg15_1, buf27, arg17_1, buf24, buf29, 1606144, grid=grid(1606144), stream=stream0)
        del buf25
        del buf26
        del buf27
        buf30 = reinterpret_tensor(buf24, (25096, 64), (64, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (25096, 64), (64, 1), 0), reinterpret_tensor(arg18_1, (64, 64), (1, 64), 0), out=buf30)
        del arg18_1
        buf31 = reinterpret_tensor(buf30, (8, 3137, 64), (200768, 64, 1), 0); del buf30  # reuse
        buf35 = reinterpret_tensor(buf29, (8, 3137, 64), (200768, 64, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_154, x_160, x_161], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_13.run(buf31, buf8, buf9, arg7_1, arg19_1, arg20_1, arg21_1, buf35, 25096, 64, grid=grid(25096), stream=stream0)
        del arg19_1
        del arg20_1
        del arg21_1
        del buf9
        buf36 = empty_strided_cuda((25096, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf35, (25096, 64), (64, 1), 0), reinterpret_tensor(arg22_1, (64, 512), (1, 64), 0), out=buf36)
        del arg22_1
        buf37 = reinterpret_tensor(buf36, (8, 3137, 512), (1606144, 512, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf37, arg23_1, 12849152, grid=grid(12849152), stream=stream0)
        del arg23_1
        buf38 = reinterpret_tensor(buf35, (25096, 64), (64, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf37, (25096, 512), (512, 1), 0), reinterpret_tensor(arg24_1, (512, 64), (1, 512), 0), out=buf38)
        del arg24_1
        buf39 = reinterpret_tensor(buf38, (8, 3137, 64), (200768, 64, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_167], Original ATen: [aten.add]
        triton_poi_fused_add_15.run(buf39, buf31, arg25_1, 1606144, grid=grid(1606144), stream=stream0)
        del arg25_1
        # Topologically Sorted Source Nodes: [conv2d_41], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(reinterpret_tensor(buf39, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf40, (8, 64, 56, 56), (200704, 1, 3584, 64))
        del arg6_1
        buf45 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_170, x_171], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_16.run(buf39, buf40, arg7_1, arg26_1, arg27_1, buf45, 25096, 64, grid=grid(25096), stream=stream0)
        del arg26_1
        del arg27_1
        buf46 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg29_1, reinterpret_tensor(buf45, (25096, 64), (64, 1), 0), reinterpret_tensor(arg28_1, (64, 192), (1, 64), 0), alpha=1, beta=1, out=buf46)
        del arg28_1
        del arg29_1
        buf47 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_9], Original ATen: [aten._softmax]
        triton_red_fused__softmax_5.run(buf46, buf47, 12800, 126, grid=grid(12800), stream=stream0)
        buf48 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_9], Original ATen: [aten._softmax]
        triton_per_fused__softmax_6.run(buf47, buf48, 512, 25, grid=grid(512), stream=stream0)
        buf49 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_9], Original ATen: [aten._softmax]
        triton_red_fused__softmax_7.run(buf46, buf48, buf49, 12800, 126, grid=grid(12800), stream=stream0)
        buf50 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_9], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf49, buf50, 512, 25, grid=grid(512), stream=stream0)
        del buf49
        buf51 = reinterpret_tensor(buf45, (8, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_9], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_9.run(buf46, buf48, buf50, buf51, 1606144, grid=grid(1606144), stream=stream0)
        del buf48
        del buf50
        buf52 = reinterpret_tensor(buf8, (8, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [factor_att_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf46, buf52, 1606144, grid=grid(1606144), stream=stream0)
        buf53 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [factor_att_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf51, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf52, (64, 3137, 8), (25096, 8, 1), 0), out=buf53)
        buf54 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [factor_att_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf46, buf54, 1606144, grid=grid(1606144), stream=stream0)
        buf55 = reinterpret_tensor(buf51, (64, 3137, 8), (25096, 8, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [factor_att_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (64, 3137, 8), (25096, 8, 1), 0), buf53, out=buf55)
        # Topologically Sorted Source Nodes: [conv2d_42], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(reinterpret_tensor(buf46, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), arg12_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf56, (8, 16, 56, 56), (50176, 1, 896, 16))
        del arg12_1
        # Topologically Sorted Source Nodes: [conv2d_43], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(reinterpret_tensor(buf46, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), arg14_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf57, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg14_1
        # Topologically Sorted Source Nodes: [conv2d_44], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(reinterpret_tensor(buf46, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), arg16_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf58, (8, 24, 56, 56), (75264, 1, 1344, 24))
        del arg16_1
        buf60 = reinterpret_tensor(buf54, (8, 3137, 8, 8), (200768, 64, 8, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [EV_hat_18, EV_hat_19, x_173], Original ATen: [aten.mul, aten.constant_pad_nd, aten.clone]
        triton_poi_fused_clone_constant_pad_nd_mul_12.run(buf46, buf56, arg13_1, buf57, arg15_1, buf58, arg17_1, buf55, buf60, 1606144, grid=grid(1606144), stream=stream0)
        del arg13_1
        del arg15_1
        del arg17_1
        del buf46
        del buf56
        del buf57
        del buf58
        buf61 = reinterpret_tensor(buf55, (25096, 64), (64, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (25096, 64), (64, 1), 0), reinterpret_tensor(arg30_1, (64, 64), (1, 64), 0), out=buf61)
        del arg30_1
        buf62 = reinterpret_tensor(buf61, (8, 3137, 64), (200768, 64, 1), 0); del buf61  # reuse
        buf66 = reinterpret_tensor(buf60, (8, 3137, 64), (200768, 64, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_170, x_176, x_177], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_17.run(buf62, buf39, buf40, arg7_1, arg31_1, arg32_1, arg33_1, buf66, 25096, 64, grid=grid(25096), stream=stream0)
        del arg31_1
        del arg32_1
        del arg33_1
        del arg7_1
        del buf39
        buf67 = reinterpret_tensor(buf37, (25096, 512), (512, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf66, (25096, 64), (64, 1), 0), reinterpret_tensor(arg34_1, (64, 512), (1, 64), 0), out=buf67)
        del arg34_1
        buf68 = reinterpret_tensor(buf67, (8, 3137, 512), (1606144, 512, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_179], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf68, arg35_1, 12849152, grid=grid(12849152), stream=stream0)
        del arg35_1
        buf69 = reinterpret_tensor(buf66, (25096, 64), (64, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (25096, 512), (512, 1), 0), reinterpret_tensor(arg36_1, (512, 64), (1, 512), 0), out=buf69)
        del arg36_1
        del buf68
        buf70 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x1_nocls_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf62, buf69, arg37_1, buf70, 1605632, grid=grid(1605632), stream=stream0)
        del arg37_1
        del buf62
        del buf69
        buf71 = empty_strided_cuda((128, 64, 2, 2), (256, 1, 128, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x1_nocls_1, x_184], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_19.run(arg38_1, buf71, 8192, 4, grid=grid(8192, 4), stream=stream0)
        del arg38_1
        # Topologically Sorted Source Nodes: [x1_nocls_1, x_184], Original ATen: [aten.clone, aten.convolution]
        buf72 = extern_kernels.convolution(buf70, buf71, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del buf70
        del buf71
        buf78 = empty_strided_cuda((8, 785, 128), (100480, 128, 1), torch.float32)
        buf77 = reinterpret_tensor(buf78, (8, 784, 128), (100480, 128, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [x_186], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_20.run(buf72, arg39_1, arg40_1, arg41_1, buf77, 6272, 128, grid=grid(6272), stream=stream0)
        del arg39_1
        del arg40_1
        del arg41_1
        del buf72
        buf76 = reinterpret_tensor(buf78, (8, 1, 128), (100480, 128, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_187], Original ATen: [aten.cat]
        triton_poi_fused_cat_21.run(arg42_1, buf76, 1024, grid=grid(1024), stream=stream0)
        del arg42_1
        del buf76
        del buf77
        # Topologically Sorted Source Nodes: [conv2d_46], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(reinterpret_tensor(buf78, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), arg43_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf79, (8, 128, 28, 28), (100352, 1, 3584, 128))
        buf84 = empty_strided_cuda((8, 785, 128), (100480, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_190, x_191], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_22.run(buf78, buf79, arg44_1, arg45_1, arg46_1, buf84, 6280, 128, grid=grid(6280), stream=stream0)
        del arg45_1
        del arg46_1
        buf85 = empty_strided_cuda((6280, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg48_1, reinterpret_tensor(buf84, (6280, 128), (128, 1), 0), reinterpret_tensor(arg47_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf85)
        del arg47_1
        del arg48_1
        buf86 = empty_strided_cuda((8, 8, 1, 16, 7), (896, 16, 7168, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [k_softmax_10], Original ATen: [aten._softmax]
        triton_red_fused__softmax_23.run(buf85, buf86, 7168, 113, grid=grid(7168), stream=stream0)
        buf87 = empty_strided_cuda((8, 8, 1, 16), (128, 16, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_softmax_10], Original ATen: [aten._softmax]
        triton_per_fused__softmax_24.run(buf86, buf87, 1024, 7, grid=grid(1024), stream=stream0)
        buf88 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_10], Original ATen: [aten._softmax]
        triton_red_fused__softmax_25.run(buf85, buf87, buf88, 7168, 113, grid=grid(7168), stream=stream0)
        buf89 = empty_strided_cuda((8, 8, 1, 16), (128, 16, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_softmax_10], Original ATen: [aten._softmax]
        triton_per_fused__softmax_26.run(buf88, buf89, 1024, 7, grid=grid(1024), stream=stream0)
        buf90 = reinterpret_tensor(buf84, (8, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_10], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_27.run(buf85, buf87, buf89, buf90, 803840, grid=grid(803840), stream=stream0)
        buf91 = empty_strided_cuda((8, 8, 785, 16), (100480, 12560, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [factor_att_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf85, buf91, 803840, grid=grid(803840), stream=stream0)
        buf92 = empty_strided_cuda((64, 16, 16), (256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [factor_att_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf91, (64, 785, 16), (12560, 16, 1), 0), out=buf92)
        buf93 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [factor_att_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf85, buf93, 803840, grid=grid(803840), stream=stream0)
        buf94 = reinterpret_tensor(buf90, (64, 785, 16), (12560, 16, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [factor_att_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf93, (64, 785, 16), (12560, 16, 1), 0), buf92, out=buf94)
        # Topologically Sorted Source Nodes: [conv2d_47], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(reinterpret_tensor(buf85, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), arg49_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf95, (8, 32, 28, 28), (25088, 1, 896, 32))
        # Topologically Sorted Source Nodes: [conv2d_48], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(reinterpret_tensor(buf85, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), arg51_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf96, (8, 48, 28, 28), (37632, 1, 1344, 48))
        # Topologically Sorted Source Nodes: [conv2d_49], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(reinterpret_tensor(buf85, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), arg53_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf97, (8, 48, 28, 28), (37632, 1, 1344, 48))
        buf99 = reinterpret_tensor(buf93, (8, 785, 8, 16), (100480, 128, 16, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [EV_hat_20, EV_hat_21, x_193], Original ATen: [aten.mul, aten.constant_pad_nd, aten.clone]
        triton_poi_fused_clone_constant_pad_nd_mul_30.run(buf85, buf95, arg50_1, buf96, arg52_1, buf97, arg54_1, buf94, buf99, 803840, grid=grid(803840), stream=stream0)
        del buf95
        del buf96
        del buf97
        buf100 = reinterpret_tensor(buf94, (6280, 128), (128, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (6280, 128), (128, 1), 0), reinterpret_tensor(arg55_1, (128, 128), (1, 128), 0), out=buf100)
        del arg55_1
        buf101 = reinterpret_tensor(buf100, (8, 785, 128), (100480, 128, 1), 0); del buf100  # reuse
        buf105 = reinterpret_tensor(buf99, (8, 785, 128), (100480, 128, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [x_190, x_196, x_197], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_31.run(buf101, buf78, buf79, arg44_1, arg56_1, arg57_1, arg58_1, buf105, 6280, 128, grid=grid(6280), stream=stream0)
        del arg56_1
        del arg57_1
        del arg58_1
        del buf79
        buf106 = empty_strided_cuda((6280, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (6280, 128), (128, 1), 0), reinterpret_tensor(arg59_1, (128, 1024), (1, 128), 0), out=buf106)
        del arg59_1
        buf107 = reinterpret_tensor(buf106, (8, 785, 1024), (803840, 1024, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_199], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_32.run(buf107, arg60_1, 6430720, grid=grid(6430720), stream=stream0)
        del arg60_1
        buf108 = reinterpret_tensor(buf105, (6280, 128), (128, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf107, (6280, 1024), (1024, 1), 0), reinterpret_tensor(arg61_1, (1024, 128), (1, 1024), 0), out=buf108)
        del arg61_1
        buf109 = reinterpret_tensor(buf108, (8, 785, 128), (100480, 128, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_203], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf109, buf101, arg62_1, 803840, grid=grid(803840), stream=stream0)
        del arg62_1
        # Topologically Sorted Source Nodes: [conv2d_50], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(reinterpret_tensor(buf109, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), arg43_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf110, (8, 128, 28, 28), (100352, 1, 3584, 128))
        del arg43_1
        buf115 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_206, x_207], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_34.run(buf109, buf110, arg44_1, arg63_1, arg64_1, buf115, 6280, 128, grid=grid(6280), stream=stream0)
        del arg63_1
        del arg64_1
        buf116 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg66_1, reinterpret_tensor(buf115, (6280, 128), (128, 1), 0), reinterpret_tensor(arg65_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf116)
        del arg65_1
        del arg66_1
        buf117 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_11], Original ATen: [aten._softmax]
        triton_red_fused__softmax_23.run(buf116, buf117, 7168, 113, grid=grid(7168), stream=stream0)
        buf118 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_24.run(buf117, buf118, 1024, 7, grid=grid(1024), stream=stream0)
        buf119 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_11], Original ATen: [aten._softmax]
        triton_red_fused__softmax_25.run(buf116, buf118, buf119, 7168, 113, grid=grid(7168), stream=stream0)
        buf120 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_26.run(buf119, buf120, 1024, 7, grid=grid(1024), stream=stream0)
        del buf119
        buf121 = reinterpret_tensor(buf115, (8, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_11], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_27.run(buf116, buf118, buf120, buf121, 803840, grid=grid(803840), stream=stream0)
        del buf118
        del buf120
        buf122 = reinterpret_tensor(buf78, (8, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [factor_att_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf116, buf122, 803840, grid=grid(803840), stream=stream0)
        buf123 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [factor_att_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf122, (64, 785, 16), (12560, 16, 1), 0), out=buf123)
        buf124 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [factor_att_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf116, buf124, 803840, grid=grid(803840), stream=stream0)
        buf125 = reinterpret_tensor(buf121, (64, 785, 16), (12560, 16, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [factor_att_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf124, (64, 785, 16), (12560, 16, 1), 0), buf123, out=buf125)
        del buf123
        # Topologically Sorted Source Nodes: [conv2d_51], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(reinterpret_tensor(buf116, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), arg49_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf126, (8, 32, 28, 28), (25088, 1, 896, 32))
        del arg49_1
        # Topologically Sorted Source Nodes: [conv2d_52], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(reinterpret_tensor(buf116, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), arg51_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf127, (8, 48, 28, 28), (37632, 1, 1344, 48))
        del arg51_1
        # Topologically Sorted Source Nodes: [conv2d_53], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(reinterpret_tensor(buf116, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), arg53_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf128, (8, 48, 28, 28), (37632, 1, 1344, 48))
        del arg53_1
        buf130 = reinterpret_tensor(buf124, (8, 785, 8, 16), (100480, 128, 16, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [EV_hat_22, EV_hat_23, x_209], Original ATen: [aten.mul, aten.constant_pad_nd, aten.clone]
        triton_poi_fused_clone_constant_pad_nd_mul_30.run(buf116, buf126, arg50_1, buf127, arg52_1, buf128, arg54_1, buf125, buf130, 803840, grid=grid(803840), stream=stream0)
        del arg50_1
        del arg52_1
        del arg54_1
        del buf116
        del buf126
        del buf127
        del buf128
        buf131 = reinterpret_tensor(buf125, (6280, 128), (128, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (6280, 128), (128, 1), 0), reinterpret_tensor(arg67_1, (128, 128), (1, 128), 0), out=buf131)
        del arg67_1
        buf132 = reinterpret_tensor(buf131, (8, 785, 128), (100480, 128, 1), 0); del buf131  # reuse
        buf136 = reinterpret_tensor(buf130, (8, 785, 128), (100480, 128, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_206, x_212, x_213], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_35.run(buf132, buf109, buf110, arg44_1, arg68_1, arg69_1, arg70_1, buf136, 6280, 128, grid=grid(6280), stream=stream0)
        del arg44_1
        del arg68_1
        del arg69_1
        del arg70_1
        del buf109
        buf137 = reinterpret_tensor(buf107, (6280, 1024), (1024, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (6280, 128), (128, 1), 0), reinterpret_tensor(arg71_1, (128, 1024), (1, 128), 0), out=buf137)
        del arg71_1
        buf138 = reinterpret_tensor(buf137, (8, 785, 1024), (803840, 1024, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [x_215], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_32.run(buf138, arg72_1, 6430720, grid=grid(6430720), stream=stream0)
        del arg72_1
        buf139 = reinterpret_tensor(buf136, (6280, 128), (128, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (6280, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 128), (1, 1024), 0), out=buf139)
        del arg73_1
        del buf138
        buf140 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x2_nocls_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_36.run(buf132, buf139, arg74_1, buf140, 802816, grid=grid(802816), stream=stream0)
        del arg74_1
        del buf132
        del buf139
        buf141 = empty_strided_cuda((320, 128, 2, 2), (512, 1, 256, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x2_nocls_1, x_220], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_37.run(arg75_1, buf141, 40960, 4, grid=grid(40960, 4), stream=stream0)
        del arg75_1
        # Topologically Sorted Source Nodes: [x2_nocls_1, x_220], Original ATen: [aten.clone, aten.convolution]
        buf142 = extern_kernels.convolution(buf140, buf141, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 320, 14, 14), (62720, 1, 4480, 320))
        del buf140
        del buf141
        buf148 = empty_strided_cuda((8, 197, 320), (63040, 320, 1), torch.float32)
        buf147 = reinterpret_tensor(buf148, (8, 196, 320), (63040, 320, 1), 320)  # alias
        # Topologically Sorted Source Nodes: [x_222], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_38.run(buf142, arg76_1, arg77_1, arg78_1, buf147, 1568, 320, grid=grid(1568), stream=stream0)
        del arg76_1
        del arg77_1
        del arg78_1
        del buf142
        buf146 = reinterpret_tensor(buf148, (8, 1, 320), (63040, 320, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_223], Original ATen: [aten.cat]
        triton_poi_fused_cat_39.run(arg79_1, buf146, 2560, grid=grid(2560), stream=stream0)
        del arg79_1
        del buf146
        del buf147
        # Topologically Sorted Source Nodes: [conv2d_55], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(reinterpret_tensor(buf148, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), arg80_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320, bias=None)
        assert_size_stride(buf149, (8, 320, 14, 14), (62720, 1, 4480, 320))
        buf154 = empty_strided_cuda((8, 197, 320), (63040, 320, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_226, x_227], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_40.run(buf148, buf149, arg81_1, arg82_1, arg83_1, buf154, 1576, 320, grid=grid(1576), stream=stream0)
        del arg82_1
        del arg83_1
        buf155 = empty_strided_cuda((1576, 960), (960, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg85_1, reinterpret_tensor(buf154, (1576, 320), (320, 1), 0), reinterpret_tensor(arg84_1, (320, 960), (1, 320), 0), alpha=1, beta=1, out=buf155)
        del arg84_1
        del arg85_1
        buf156 = empty_strided_cuda((8, 8, 1, 40, 2), (640, 40, 5120, 1, 320), torch.float32)
        # Topologically Sorted Source Nodes: [k_softmax_12], Original ATen: [aten._softmax]
        triton_red_fused__softmax_41.run(buf155, buf156, 5120, 99, grid=grid(5120), stream=stream0)
        buf157 = empty_strided_cuda((8, 8, 1, 40), (320, 40, 2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_softmax_12], Original ATen: [aten._softmax]
        triton_per_fused__softmax_42.run(buf156, buf157, 2560, 2, grid=grid(2560), stream=stream0)
        buf158 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_12], Original ATen: [aten._softmax]
        triton_red_fused__softmax_43.run(buf155, buf157, buf158, 5120, 99, grid=grid(5120), stream=stream0)
        buf159 = empty_strided_cuda((8, 8, 1, 40), (320, 40, 2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_softmax_12], Original ATen: [aten._softmax]
        triton_per_fused__softmax_44.run(buf158, buf159, 2560, 2, grid=grid(2560), stream=stream0)
        buf160 = reinterpret_tensor(buf154, (8, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_12], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_45.run(buf155, buf157, buf159, buf160, 504320, grid=grid(504320), stream=stream0)
        buf161 = empty_strided_cuda((8, 8, 197, 40), (63040, 7880, 40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [factor_att_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_46.run(buf155, buf161, 504320, grid=grid(504320), stream=stream0)
        buf162 = empty_strided_cuda((64, 40, 40), (1600, 40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [factor_att_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf160, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf161, (64, 197, 40), (7880, 40, 1), 0), out=buf162)
        buf163 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [factor_att_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_47.run(buf155, buf163, 504320, grid=grid(504320), stream=stream0)
        buf164 = reinterpret_tensor(buf160, (64, 197, 40), (7880, 40, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [factor_att_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf163, (64, 197, 40), (7880, 40, 1), 0), buf162, out=buf164)
        # Topologically Sorted Source Nodes: [conv2d_56], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(reinterpret_tensor(buf155, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), arg86_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf165, (8, 80, 14, 14), (15680, 1, 1120, 80))
        # Topologically Sorted Source Nodes: [conv2d_57], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(reinterpret_tensor(buf155, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), arg88_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf166, (8, 120, 14, 14), (23520, 1, 1680, 120))
        # Topologically Sorted Source Nodes: [conv2d_58], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(reinterpret_tensor(buf155, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), arg90_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf167, (8, 120, 14, 14), (23520, 1, 1680, 120))
        buf169 = reinterpret_tensor(buf163, (8, 197, 8, 40), (63040, 320, 40, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [EV_hat_24, EV_hat_25, x_229], Original ATen: [aten.mul, aten.constant_pad_nd, aten.clone]
        triton_poi_fused_clone_constant_pad_nd_mul_48.run(buf155, buf165, arg87_1, buf166, arg89_1, buf167, arg91_1, buf164, buf169, 504320, grid=grid(504320), stream=stream0)
        del buf165
        del buf166
        del buf167
        buf170 = reinterpret_tensor(buf164, (1576, 320), (320, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (1576, 320), (320, 1), 0), reinterpret_tensor(arg92_1, (320, 320), (1, 320), 0), out=buf170)
        del arg92_1
        buf171 = reinterpret_tensor(buf170, (8, 197, 320), (63040, 320, 1), 0); del buf170  # reuse
        buf175 = reinterpret_tensor(buf169, (8, 197, 320), (63040, 320, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_226, x_232, x_233], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_49.run(buf171, buf148, buf149, arg81_1, arg93_1, arg94_1, arg95_1, buf175, 1576, 320, grid=grid(1576), stream=stream0)
        del arg93_1
        del arg94_1
        del arg95_1
        del buf149
        buf176 = empty_strided_cuda((1576, 1280), (1280, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf175, (1576, 320), (320, 1), 0), reinterpret_tensor(arg96_1, (320, 1280), (1, 320), 0), out=buf176)
        del arg96_1
        buf177 = reinterpret_tensor(buf176, (8, 197, 1280), (252160, 1280, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [x_235], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_50.run(buf177, arg97_1, 2017280, grid=grid(2017280), stream=stream0)
        del arg97_1
        buf178 = reinterpret_tensor(buf175, (1576, 320), (320, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf177, (1576, 1280), (1280, 1), 0), reinterpret_tensor(arg98_1, (1280, 320), (1, 1280), 0), out=buf178)
        del arg98_1
        buf179 = reinterpret_tensor(buf178, (8, 197, 320), (63040, 320, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_239], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(buf179, buf171, arg99_1, 504320, grid=grid(504320), stream=stream0)
        del arg99_1
        # Topologically Sorted Source Nodes: [conv2d_59], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(reinterpret_tensor(buf179, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), arg80_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320, bias=None)
        assert_size_stride(buf180, (8, 320, 14, 14), (62720, 1, 4480, 320))
        del arg80_1
        buf185 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_242, x_243], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_52.run(buf179, buf180, arg81_1, arg100_1, arg101_1, buf185, 1576, 320, grid=grid(1576), stream=stream0)
        del arg100_1
        del arg101_1
        buf186 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg103_1, reinterpret_tensor(buf185, (1576, 320), (320, 1), 0), reinterpret_tensor(arg102_1, (320, 960), (1, 320), 0), alpha=1, beta=1, out=buf186)
        del arg102_1
        del arg103_1
        buf187 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_13], Original ATen: [aten._softmax]
        triton_red_fused__softmax_41.run(buf186, buf187, 5120, 99, grid=grid(5120), stream=stream0)
        buf188 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_13], Original ATen: [aten._softmax]
        triton_per_fused__softmax_42.run(buf187, buf188, 2560, 2, grid=grid(2560), stream=stream0)
        buf189 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_13], Original ATen: [aten._softmax]
        triton_red_fused__softmax_43.run(buf186, buf188, buf189, 5120, 99, grid=grid(5120), stream=stream0)
        buf190 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_13], Original ATen: [aten._softmax]
        triton_per_fused__softmax_44.run(buf189, buf190, 2560, 2, grid=grid(2560), stream=stream0)
        del buf189
        buf191 = reinterpret_tensor(buf185, (8, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_13], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_45.run(buf186, buf188, buf190, buf191, 504320, grid=grid(504320), stream=stream0)
        del buf188
        del buf190
        buf192 = reinterpret_tensor(buf148, (8, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [factor_att_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_46.run(buf186, buf192, 504320, grid=grid(504320), stream=stream0)
        buf193 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [factor_att_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf191, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf192, (64, 197, 40), (7880, 40, 1), 0), out=buf193)
        buf194 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [factor_att_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_47.run(buf186, buf194, 504320, grid=grid(504320), stream=stream0)
        buf195 = reinterpret_tensor(buf191, (64, 197, 40), (7880, 40, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [factor_att_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf194, (64, 197, 40), (7880, 40, 1), 0), buf193, out=buf195)
        del buf193
        # Topologically Sorted Source Nodes: [conv2d_60], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(reinterpret_tensor(buf186, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), arg86_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf196, (8, 80, 14, 14), (15680, 1, 1120, 80))
        del arg86_1
        # Topologically Sorted Source Nodes: [conv2d_61], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(reinterpret_tensor(buf186, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), arg88_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf197, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg88_1
        # Topologically Sorted Source Nodes: [conv2d_62], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(reinterpret_tensor(buf186, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), arg90_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf198, (8, 120, 14, 14), (23520, 1, 1680, 120))
        del arg90_1
        buf200 = reinterpret_tensor(buf194, (8, 197, 8, 40), (63040, 320, 40, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [EV_hat_26, EV_hat_27, x_245], Original ATen: [aten.mul, aten.constant_pad_nd, aten.clone]
        triton_poi_fused_clone_constant_pad_nd_mul_48.run(buf186, buf196, arg87_1, buf197, arg89_1, buf198, arg91_1, buf195, buf200, 504320, grid=grid(504320), stream=stream0)
        del arg87_1
        del arg89_1
        del arg91_1
        del buf186
        del buf196
        del buf197
        del buf198
        buf201 = reinterpret_tensor(buf195, (1576, 320), (320, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (1576, 320), (320, 1), 0), reinterpret_tensor(arg104_1, (320, 320), (1, 320), 0), out=buf201)
        del arg104_1
        buf202 = reinterpret_tensor(buf201, (8, 197, 320), (63040, 320, 1), 0); del buf201  # reuse
        buf206 = reinterpret_tensor(buf200, (8, 197, 320), (63040, 320, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [x_242, x_248, x_249], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_53.run(buf202, buf179, buf180, arg81_1, arg105_1, arg106_1, arg107_1, buf206, 1576, 320, grid=grid(1576), stream=stream0)
        del arg105_1
        del arg106_1
        del arg107_1
        del arg81_1
        del buf179
        buf207 = reinterpret_tensor(buf177, (1576, 1280), (1280, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (1576, 320), (320, 1), 0), reinterpret_tensor(arg108_1, (320, 1280), (1, 320), 0), out=buf207)
        del arg108_1
        buf208 = reinterpret_tensor(buf207, (8, 197, 1280), (252160, 1280, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_251], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_50.run(buf208, arg109_1, 2017280, grid=grid(2017280), stream=stream0)
        del arg109_1
        buf209 = reinterpret_tensor(buf206, (1576, 320), (320, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf208, (1576, 1280), (1280, 1), 0), reinterpret_tensor(arg110_1, (1280, 320), (1, 1280), 0), out=buf209)
        del arg110_1
        del buf208
        buf210 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [x3_nocls_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_54.run(buf202, buf209, arg111_1, buf210, 501760, grid=grid(501760), stream=stream0)
        del arg111_1
        del buf202
        del buf209
        buf211 = empty_strided_cuda((512, 320, 2, 2), (1280, 1, 640, 320), torch.float32)
        # Topologically Sorted Source Nodes: [x3_nocls_1, x_256], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_55.run(arg112_1, buf211, 163840, 4, grid=grid(163840, 4), stream=stream0)
        del arg112_1
        # Topologically Sorted Source Nodes: [x3_nocls_1, x_256], Original ATen: [aten.clone, aten.convolution]
        buf212 = extern_kernels.convolution(buf210, buf211, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (8, 512, 7, 7), (25088, 1, 3584, 512))
        del buf210
        del buf211
        buf218 = empty_strided_cuda((8, 50, 512), (25600, 512, 1), torch.float32)
        buf217 = reinterpret_tensor(buf218, (8, 49, 512), (25600, 512, 1), 512)  # alias
        # Topologically Sorted Source Nodes: [x_258], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_56.run(buf212, arg113_1, arg114_1, arg115_1, buf217, 392, 512, grid=grid(392), stream=stream0)
        del arg113_1
        del arg114_1
        del arg115_1
        del buf212
        buf216 = reinterpret_tensor(buf218, (8, 1, 512), (25600, 512, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_259], Original ATen: [aten.cat]
        triton_poi_fused_cat_57.run(arg116_1, buf216, 4096, grid=grid(4096), stream=stream0)
        del arg116_1
        del buf216
        del buf217
        # Topologically Sorted Source Nodes: [conv2d_64], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(reinterpret_tensor(buf218, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf219, (8, 512, 7, 7), (25088, 1, 3584, 512))
        buf224 = empty_strided_cuda((8, 50, 512), (25600, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_262, x_263], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_58.run(buf218, buf219, arg118_1, arg119_1, arg120_1, buf224, 400, 512, grid=grid(400), stream=stream0)
        del arg119_1
        del arg120_1
        buf225 = empty_strided_cuda((400, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg122_1, reinterpret_tensor(buf224, (400, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf225)
        del arg121_1
        del arg122_1
        buf226 = reinterpret_tensor(buf53, (8, 8, 1, 64), (512, 64, 4096, 1), 0); del buf53  # reuse
        buf227 = empty_strided_cuda((8, 8, 1, 64), (512, 64, 4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_softmax_14], Original ATen: [aten._softmax]
        triton_per_fused__softmax_59.run(buf225, buf226, buf227, 4096, 50, grid=grid(4096), stream=stream0)
        buf228 = reinterpret_tensor(buf224, (8, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_14], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_60.run(buf225, buf226, buf227, buf228, 204800, grid=grid(204800), stream=stream0)
        buf229 = empty_strided_cuda((8, 8, 50, 64), (25600, 3200, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [factor_att_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_61.run(buf225, buf229, 204800, grid=grid(204800), stream=stream0)
        buf230 = empty_strided_cuda((64, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [factor_att_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf228, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf229, (64, 50, 64), (3200, 64, 1), 0), out=buf230)
        buf231 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [factor_att_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_62.run(buf225, buf231, 204800, grid=grid(204800), stream=stream0)
        buf232 = reinterpret_tensor(buf228, (64, 50, 64), (3200, 64, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [factor_att_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf231, (64, 50, 64), (3200, 64, 1), 0), buf230, out=buf232)
        # Topologically Sorted Source Nodes: [conv2d_65], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(reinterpret_tensor(buf225, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), arg123_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf233, (8, 128, 7, 7), (6272, 1, 896, 128))
        # Topologically Sorted Source Nodes: [conv2d_66], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(reinterpret_tensor(buf225, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), arg125_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf234, (8, 192, 7, 7), (9408, 1, 1344, 192))
        # Topologically Sorted Source Nodes: [conv2d_67], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(reinterpret_tensor(buf225, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), arg127_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf235, (8, 192, 7, 7), (9408, 1, 1344, 192))
        buf237 = reinterpret_tensor(buf231, (8, 50, 8, 64), (25600, 512, 64, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [EV_hat_28, EV_hat_29, x_265], Original ATen: [aten.mul, aten.constant_pad_nd, aten.clone]
        triton_poi_fused_clone_constant_pad_nd_mul_63.run(buf225, buf233, arg124_1, buf234, arg126_1, buf235, arg128_1, buf232, buf237, 204800, grid=grid(204800), stream=stream0)
        del buf233
        del buf234
        del buf235
        buf238 = reinterpret_tensor(buf232, (400, 512), (512, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf237, (400, 512), (512, 1), 0), reinterpret_tensor(arg129_1, (512, 512), (1, 512), 0), out=buf238)
        del arg129_1
        buf239 = reinterpret_tensor(buf238, (8, 50, 512), (25600, 512, 1), 0); del buf238  # reuse
        buf243 = reinterpret_tensor(buf237, (8, 50, 512), (25600, 512, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [x_262, x_268, x_269], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_64.run(buf239, buf218, buf219, arg118_1, arg130_1, arg131_1, arg132_1, buf243, 400, 512, grid=grid(400), stream=stream0)
        del arg130_1
        del arg131_1
        del arg132_1
        del buf219
        buf244 = empty_strided_cuda((400, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (400, 512), (512, 1), 0), reinterpret_tensor(arg133_1, (512, 2048), (1, 512), 0), out=buf244)
        del arg133_1
        buf245 = reinterpret_tensor(buf244, (8, 50, 2048), (102400, 2048, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [x_271], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_65.run(buf245, arg134_1, 819200, grid=grid(819200), stream=stream0)
        del arg134_1
        buf246 = reinterpret_tensor(buf243, (400, 512), (512, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (400, 2048), (2048, 1), 0), reinterpret_tensor(arg135_1, (2048, 512), (1, 2048), 0), out=buf246)
        del arg135_1
        buf247 = reinterpret_tensor(buf246, (8, 50, 512), (25600, 512, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [x_275], Original ATen: [aten.add]
        triton_poi_fused_add_66.run(buf247, buf239, arg136_1, 204800, grid=grid(204800), stream=stream0)
        del arg136_1
        # Topologically Sorted Source Nodes: [conv2d_68], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(reinterpret_tensor(buf247, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf248, (8, 512, 7, 7), (25088, 1, 3584, 512))
        del arg117_1
        buf253 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [x_278, x_279], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_67.run(buf247, buf248, arg118_1, arg137_1, arg138_1, buf253, 400, 512, grid=grid(400), stream=stream0)
        del arg137_1
        del arg138_1
        buf254 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg140_1, reinterpret_tensor(buf253, (400, 512), (512, 1), 0), reinterpret_tensor(arg139_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf254)
        del arg139_1
        del arg140_1
        buf255 = buf227; del buf227  # reuse
        buf256 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_59.run(buf254, buf255, buf256, 4096, 50, grid=grid(4096), stream=stream0)
        buf257 = reinterpret_tensor(buf253, (8, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [k_softmax_15], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_60.run(buf254, buf255, buf256, buf257, 204800, grid=grid(204800), stream=stream0)
        del buf255
        buf258 = reinterpret_tensor(buf218, (8, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [factor_att_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_61.run(buf254, buf258, 204800, grid=grid(204800), stream=stream0)
        buf259 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [factor_att_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf257, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf258, (64, 50, 64), (3200, 64, 1), 0), out=buf259)
        buf260 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [factor_att_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_62.run(buf254, buf260, 204800, grid=grid(204800), stream=stream0)
        buf261 = reinterpret_tensor(buf257, (64, 50, 64), (3200, 64, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [factor_att_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf260, (64, 50, 64), (3200, 64, 1), 0), buf259, out=buf261)
        del buf259
        # Topologically Sorted Source Nodes: [conv2d_69], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(reinterpret_tensor(buf254, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), arg123_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf262, (8, 128, 7, 7), (6272, 1, 896, 128))
        del arg123_1
        # Topologically Sorted Source Nodes: [conv2d_70], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(reinterpret_tensor(buf254, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), arg125_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf263, (8, 192, 7, 7), (9408, 1, 1344, 192))
        del arg125_1
        # Topologically Sorted Source Nodes: [conv2d_71], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(reinterpret_tensor(buf254, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), arg127_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf264, (8, 192, 7, 7), (9408, 1, 1344, 192))
        del arg127_1
        buf266 = reinterpret_tensor(buf260, (8, 50, 8, 64), (25600, 512, 64, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [EV_hat_30, EV_hat_31, x_281], Original ATen: [aten.mul, aten.constant_pad_nd, aten.clone]
        triton_poi_fused_clone_constant_pad_nd_mul_63.run(buf254, buf262, arg124_1, buf263, arg126_1, buf264, arg128_1, buf261, buf266, 204800, grid=grid(204800), stream=stream0)
        del arg124_1
        del arg126_1
        del arg128_1
        del buf254
        del buf262
        del buf263
        del buf264
        buf267 = reinterpret_tensor(buf261, (400, 512), (512, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf266, (400, 512), (512, 1), 0), reinterpret_tensor(arg141_1, (512, 512), (1, 512), 0), out=buf267)
        del arg141_1
        buf268 = reinterpret_tensor(buf267, (8, 50, 512), (25600, 512, 1), 0); del buf267  # reuse
        buf272 = reinterpret_tensor(buf266, (8, 50, 512), (25600, 512, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [x_278, x_284, x_285], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_68.run(buf268, buf247, buf248, arg118_1, arg142_1, arg143_1, arg144_1, buf272, 400, 512, grid=grid(400), stream=stream0)
        del arg118_1
        del arg142_1
        del arg143_1
        del arg144_1
        del buf247
        del buf248
        buf273 = reinterpret_tensor(buf245, (400, 2048), (2048, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (400, 512), (512, 1), 0), reinterpret_tensor(arg145_1, (512, 2048), (1, 512), 0), out=buf273)
        del arg145_1
        buf274 = reinterpret_tensor(buf273, (8, 50, 2048), (102400, 2048, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [x_287], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_65.run(buf274, arg146_1, 819200, grid=grid(819200), stream=stream0)
        del arg146_1
        buf275 = reinterpret_tensor(buf272, (400, 512), (512, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (400, 2048), (2048, 1), 0), reinterpret_tensor(arg147_1, (2048, 512), (1, 2048), 0), out=buf275)
        del arg147_1
        del buf274
        buf276 = empty_strided_cuda((8, 50, 1), (50, 1, 400), torch.float32)
        buf277 = empty_strided_cuda((8, 50, 1), (50, 1, 400), torch.float32)
        # Topologically Sorted Source Nodes: [x_291, x_292], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_69.run(buf268, buf275, arg148_1, buf276, buf277, 400, 512, grid=grid(400), stream=stream0)
        buf279 = reinterpret_tensor(buf256, (8, 512), (512, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [x_294], Original ATen: [aten.clone]
        triton_poi_fused_clone_70.run(buf268, buf275, arg148_1, buf276, buf277, arg149_1, arg150_1, buf279, 4096, grid=grid(4096), stream=stream0)
        del arg148_1
        del arg149_1
        del arg150_1
        del buf268
        del buf275
        del buf276
        del buf277
        buf280 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_294, x_295], Original ATen: [aten.clone, aten.addmm]
        extern_kernels.addmm(arg152_1, buf279, reinterpret_tensor(arg151_1, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf280)
        del arg151_1
        del arg152_1
        del buf279
    return (buf280, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((24, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((48, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((320, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1, 1, 320), (320, 320, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((960, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((960, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1, 1, 512), (512, 512, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((192, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('coat_lite_mini', benchmark_compiled_module)
