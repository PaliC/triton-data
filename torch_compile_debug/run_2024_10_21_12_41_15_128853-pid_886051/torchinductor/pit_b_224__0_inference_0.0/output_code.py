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


# kernel path: /tmp/torchinductor_sahanp/xm/cxmtvonsarpyyzmqixjecenrp2sn74uq2scwykupeie7rhiatrj7.py
# Topologically Sorted Source Nodes: [x_165, layer_norm_27], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_27 => var_mean_27
#   x_165 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %permute_87], 1), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_3, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_cat_native_layer_norm_0 = async_compile.triton('triton_red_fused_cat_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 15392
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2) % 962
    x0 = xindex % 2
    x2 = (xindex // 1924)
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex % 1924
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = x1
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r3 + (128*x0)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 962, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((961*r3) + (123008*x0) + (246016*x2) + (((-1) + x1) % 961)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (r3 + (128*x0)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp12 = tl.load(in_ptr3 + ((961*r3) + (123008*x0) + (((-1) + x1) % 961)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp6, tmp13, tmp14)
        tmp16 = tl.where(tmp4, tmp5, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight, roffset == 0
        )
        tmp18_mean = tl.where(rmask & xmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask & xmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask & xmask, tmp18_weight_next, tmp18_weight)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tl.store(out_ptr0 + (x5 + (1952*x2)), tmp18, xmask)
    tl.store(out_ptr1 + (x5 + (1952*x2)), tmp19, xmask)
    tl.store(out_ptr2 + (x5 + (1952*x2)), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wd/cwdn5kelqhucclx5mgxfi7fcy5giab26fr6eox2la4beh2y6akka.py
# Topologically Sorted Source Nodes: [x_165, layer_norm_27], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_27 => var_mean_27
#   x_165 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %permute_87], 1), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_3, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_cat_native_layer_norm_1 = async_compile.triton('triton_per_fused_cat_native_layer_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7696
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 962
    x1 = (xindex // 962)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (2*x0) + (1952*x1)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (2*x0) + (1952*x1)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2 + (2*x0) + (1952*x1)), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uu/cuu2g7ysirf6qk5s23zso6txwjophwq4y6iez2shnmsjzxjxcnpx.py
# Topologically Sorted Source Nodes: [x_165, layer_norm_27], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_27 => add_97, add_98, mul_93, mul_94, rsqrt_27, sub_27, var_mean_27
#   x_165 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %permute_87], 1), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_3, %getitem_146), kwargs = {})
#   %add_97 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_145, 1e-06), kwargs = {})
#   %rsqrt_27 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_97,), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_27), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, %arg5_1), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %arg6_1), kwargs = {})
triton_poi_fused_cat_native_layer_norm_2 = async_compile.triton('triton_poi_fused_cat_native_layer_norm_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1970176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 256) % 962
    x0 = xindex % 256
    x2 = (xindex // 246272)
    x3 = (xindex // 256)
    x4 = xindex
    tmp17 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 962, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((961*x0) + (246016*x2) + (((-1) + x1) % 961)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr3 + ((961*x0) + (((-1) + x1) % 961)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 - tmp17
    tmp20 = 256.0
    tmp21 = tmp19 / tmp20
    tmp22 = 1e-06
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp18 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr1 + (x4), tmp29, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5o/c5oqxxfyahek3kmv5e4exxogsui74rdtftmxxv7lv2ukcpvbetqz.py
# Topologically Sorted Source Nodes: [x_165, x_170, layer_norm_28], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_28 => add_100, add_101, mul_95, mul_96, rsqrt_28, sub_28, var_mean_28
#   x_165 => cat_3
#   x_170 => add_99
# Graph fragment:
#   %cat_3 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %permute_87], 1), kwargs = {})
#   %add_99 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_3, %view_146), kwargs = {})
#   %var_mean_28 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_99, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_99, %getitem_155), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_154, 1e-06), kwargs = {})
#   %rsqrt_28 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_100,), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_28), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_95, %arg11_1), kwargs = {})
#   %add_101 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_96, %arg12_1), kwargs = {})
triton_red_fused_add_cat_native_layer_norm_3 = async_compile.triton('triton_red_fused_add_cat_native_layer_norm_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_layer_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7696
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 962
    x1 = (xindex // 962)
    x3 = xindex
    tmp22_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp17 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 962, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((961*r2) + (246016*x1) + (((-1) + x0) % 961)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp12 = tl.load(in_ptr3 + ((961*r2) + (((-1) + x0) % 961)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp6, tmp13, tmp14)
        tmp16 = tl.where(tmp4, tmp5, tmp15)
        tmp19 = tmp17 + tmp18
        tmp20 = tmp16 + tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp22_mean_next, tmp22_m2_next, tmp22_weight_next = triton_helpers.welford_reduce(
            tmp21, tmp22_mean, tmp22_m2, tmp22_weight, roffset == 0
        )
        tmp22_mean = tl.where(rmask & xmask, tmp22_mean_next, tmp22_mean)
        tmp22_m2 = tl.where(rmask & xmask, tmp22_m2_next, tmp22_m2)
        tmp22_weight = tl.where(rmask & xmask, tmp22_weight_next, tmp22_weight)
        tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp20, rmask & xmask)
    tmp22_tmp, tmp23_tmp, tmp24_tmp = triton_helpers.welford(
        tmp22_mean, tmp22_m2, tmp22_weight, 1
    )
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp25 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tmp25 - tmp22
        tmp27 = 256.0
        tmp28 = tmp23 / tmp27
        tmp29 = 1e-06
        tmp30 = tmp28 + tmp29
        tmp31 = libdevice.rsqrt(tmp30)
        tmp32 = tmp26 * tmp31
        tmp34 = tmp32 * tmp33
        tmp36 = tmp34 + tmp35
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp36, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gx/cgxaiv23fau2bljn6mg3nyr3jcjuacgnif2p2wkuqnsbkf7numf6.py
# Topologically Sorted Source Nodes: [x_172], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_172 => add_102, erf_13, mul_97, mul_98, mul_99
# Graph fragment:
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_148, 0.5), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_148, 0.7071067811865476), kwargs = {})
#   %erf_13 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_98,), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_13, 1), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_97, %add_102), kwargs = {})
triton_poi_fused_gelu_4 = async_compile.triton('triton_poi_fused_gelu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7880704
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


# kernel path: /tmp/torchinductor_sahanp/fq/cfq6becsownf2vng6mylvam5vks3sdscjhgj3pt6tfwaihxogcuz.py
# Topologically Sorted Source Nodes: [x_176, layer_norm_29], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_29 => add_104, add_105, mul_100, mul_101, rsqrt_29, sub_29, var_mean_29
#   x_176 => add_103
# Graph fragment:
#   %add_103 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_99, %view_150), kwargs = {})
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_103, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_103, %getitem_157), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_156, 1e-06), kwargs = {})
#   %rsqrt_29 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_104,), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %rsqrt_29), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %arg17_1), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %arg18_1), kwargs = {})
triton_per_fused_add_native_layer_norm_5 = async_compile.triton('triton_per_fused_add_native_layer_norm_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 7696
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 256, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp4 - tmp12
    tmp19 = 256.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hd/chdlmrfdvy3uacib7w5tj3cmkictzwqldhpedvodogv6enmcvtua.py
# Topologically Sorted Source Nodes: [x_176, x_181, layer_norm_30], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_30 => add_107, add_108, mul_102, mul_103, rsqrt_30, sub_30, var_mean_30
#   x_176 => add_103
#   x_181 => add_106
# Graph fragment:
#   %add_103 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_99, %view_150), kwargs = {})
#   %add_106 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_103, %view_156), kwargs = {})
#   %var_mean_30 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_106, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_106, %getitem_166), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_165, 1e-06), kwargs = {})
#   %rsqrt_30 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_107,), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %rsqrt_30), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, %arg23_1), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_103, %arg24_1), kwargs = {})
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_per_fused_add_native_layer_norm_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 7696
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r1 + (256*x0)), None)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tl.full([1], 256, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp9 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tmp8 - tmp16
    tmp23 = 256.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp8, None)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ne/cnepsxfwkro3v6c4gtewe27s252dupezkg2fd6zeocmsmks5ypgy.py
# Topologically Sorted Source Nodes: [x_198], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_198 => add_117
# Graph fragment:
#   %add_117 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_113, %view_170), kwargs = {})
triton_poi_fused_add_7 = async_compile.triton('triton_poi_fused_add_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1970176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6e/c6e34wrjhdgb4gpjpyzyhpsl2lb6ihytw2q2tlusnb3ai34jpdo3.py
# Topologically Sorted Source Nodes: [x_203, layer_norm_33], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_33 => add_119, add_120, mul_114, mul_115, rsqrt_33, sub_33, var_mean_33
#   x_203 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_118, %permute_108], 1), kwargs = {})
#   %var_mean_33 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_4, %getitem_179), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_178, 1e-06), kwargs = {})
#   %rsqrt_33 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_119,), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %rsqrt_33), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_114, %arg45_1), kwargs = {})
#   %add_120 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_115, %arg46_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_8 = async_compile.triton('triton_per_fused_cat_native_layer_norm_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, xnumel, rnumel):
    xnumel = 2056
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex % 257
    r2 = rindex
    x1 = (xindex // 257)
    x3 = xindex
    tmp39 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (512*x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 257, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (r2 + (512*(((-1) + x0) % 256)) + (131072*x1)), tmp10, other=0.0)
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), tmp10, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp19 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = tmp18 - tmp26
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-06
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp42, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cr/ccrl2vyqipdobwalytohjaf236g3k75fugmvfgebdo7wn4nkovf6.py
# Topologically Sorted Source Nodes: [x_203, x_208, layer_norm_34], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_34 => add_122, add_123, mul_116, mul_117, rsqrt_34, sub_34, var_mean_34
#   x_203 => cat_4
#   x_208 => add_121
# Graph fragment:
#   %cat_4 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_118, %permute_108], 1), kwargs = {})
#   %add_121 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_4, %view_180), kwargs = {})
#   %var_mean_34 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_121, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_121, %getitem_188), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_187, 1e-06), kwargs = {})
#   %rsqrt_34 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_122,), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %rsqrt_34), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_116, %arg51_1), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_117, %arg52_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_9 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 2056
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex % 257
    r2 = rindex
    x1 = (xindex // 257)
    x3 = xindex
    tmp19 = tl.load(in_out_ptr0 + (r2 + (512*x3)), None)
    tmp20 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr6 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (512*x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 257, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (r2 + (512*(((-1) + x0) % 256)) + (131072*x1)), tmp10, other=0.0)
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), tmp10, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = tl.full([1], 512, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp23 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp36 = tmp22 - tmp30
    tmp37 = 512.0
    tmp38 = tmp35 / tmp37
    tmp39 = 1e-06
    tmp40 = tmp38 + tmp39
    tmp41 = libdevice.rsqrt(tmp40)
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp22, None)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp46, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/iz/cizetzvondrvhnj32xsg3rivrdkdkdft5vbnik2u4pqnvdjset7j.py
# Topologically Sorted Source Nodes: [x_210], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_210 => add_124, erf_16, mul_118, mul_119, mul_120
# Graph fragment:
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_182, 0.5), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_182, 0.7071067811865476), kwargs = {})
#   %erf_16 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_119,), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_16, 1), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %add_124), kwargs = {})
triton_poi_fused_gelu_10 = async_compile.triton('triton_poi_fused_gelu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4210688
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


# kernel path: /tmp/torchinductor_sahanp/ri/cri52jlcre63wmrfjmrudprxky3bycxynh5dzl7mjx2qgyx4wt65.py
# Topologically Sorted Source Nodes: [x_214, layer_norm_35], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_35 => add_126, add_127, mul_121, mul_122, rsqrt_35, sub_35, var_mean_35
#   x_214 => add_125
# Graph fragment:
#   %add_125 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_121, %view_184), kwargs = {})
#   %var_mean_35 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_125, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_125, %getitem_190), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_189, 1e-06), kwargs = {})
#   %rsqrt_35 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_126,), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %rsqrt_35), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_121, %arg57_1), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_122, %arg58_1), kwargs = {})
triton_per_fused_add_native_layer_norm_11 = async_compile.triton('triton_per_fused_add_native_layer_norm_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 2056
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
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
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
    tmp18 = tmp4 - tmp12
    tmp19 = 512.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/u5/cu5ppnbqpgtunacsdopngdljerntjpmquz7s6codnmfgz5zfepxv.py
# Topologically Sorted Source Nodes: [x_214, x_219, layer_norm_36], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_36 => add_129, add_130, mul_123, mul_124, rsqrt_36, sub_36, var_mean_36
#   x_214 => add_125
#   x_219 => add_128
# Graph fragment:
#   %add_125 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_121, %view_184), kwargs = {})
#   %add_128 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_125, %view_190), kwargs = {})
#   %var_mean_36 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_128, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_128, %getitem_199), kwargs = {})
#   %add_129 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_198, 1e-06), kwargs = {})
#   %rsqrt_36 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_129,), kwargs = {})
#   %mul_123 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %rsqrt_36), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_123, %arg63_1), kwargs = {})
#   %add_130 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_124, %arg64_1), kwargs = {})
triton_per_fused_add_native_layer_norm_12 = async_compile.triton('triton_per_fused_add_native_layer_norm_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 2056
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
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), None)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tl.full([1], 512, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp9 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tmp8 - tmp16
    tmp23 = 512.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, None)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pk/cpkmaa2ewdzzw5gwwb2vcleqrksiwmayb7y7clc7xrtxqmmrzy5b.py
# Topologically Sorted Source Nodes: [x_269], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_269 => add_160
# Graph fragment:
#   %add_160 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_156, %view_234), kwargs = {})
triton_poi_fused_add_13 = async_compile.triton('triton_poi_fused_add_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_13(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1052672
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


# kernel path: /tmp/torchinductor_sahanp/qp/cqpygkagpzyvfx7o4awv6lowviq3xi5sywikpord6rr4334kpzqk.py
# Topologically Sorted Source Nodes: [x_274, layer_norm_45], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_45 => add_162, add_163, mul_156, mul_157, rsqrt_45, sub_45, var_mean_45
#   x_274 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_161, %permute_147], 1), kwargs = {})
#   %var_mean_45 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_5, %getitem_245), kwargs = {})
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_244, 1e-06), kwargs = {})
#   %rsqrt_45 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_162,), kwargs = {})
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %rsqrt_45), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_156, %arg121_1), kwargs = {})
#   %add_163 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_157, %arg122_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_14 = async_compile.triton('triton_per_fused_cat_native_layer_norm_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, xnumel, rnumel):
    xnumel = 520
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex % 65
    r2 = rindex
    x1 = (xindex // 65)
    x3 = xindex
    tmp39 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (1024*x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 65, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (r2 + (1024*(((-1) + x0) % 64)) + (65536*x1)), tmp10, other=0.0)
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), tmp10, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tl.full([1], 1024, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp19 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = tmp18 - tmp26
    tmp33 = 1024.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-06
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tl.store(out_ptr3 + (r2 + (1024*x3)), tmp42, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3l/c3lpojivkf5vq73nwmb2kxg6s32ldfzwvyxnx2vyqsvmozybtxrk.py
# Topologically Sorted Source Nodes: [x_274, x_279, layer_norm_46], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_46 => add_165, add_166, mul_158, mul_159, rsqrt_46, sub_46, var_mean_46
#   x_274 => cat_5
#   x_279 => add_164
# Graph fragment:
#   %cat_5 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_161, %permute_147], 1), kwargs = {})
#   %add_164 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_5, %view_244), kwargs = {})
#   %var_mean_46 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_164, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_164, %getitem_254), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_253, 1e-06), kwargs = {})
#   %rsqrt_46 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_165,), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %rsqrt_46), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_158, %arg127_1), kwargs = {})
#   %add_166 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_159, %arg128_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_15 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 520
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex % 65
    r2 = rindex
    x1 = (xindex // 65)
    x3 = xindex
    tmp19 = tl.load(in_out_ptr0 + (r2 + (1024*x3)), None)
    tmp20 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr6 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (1024*x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 65, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + (r2 + (1024*(((-1) + x0) % 64)) + (65536*x1)), tmp10, other=0.0)
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), tmp10, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = tl.full([1], 1024, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp23 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp36 = tmp22 - tmp30
    tmp37 = 1024.0
    tmp38 = tmp35 / tmp37
    tmp39 = 1e-06
    tmp40 = tmp38 + tmp39
    tmp41 = libdevice.rsqrt(tmp40)
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tl.store(in_out_ptr0 + (r2 + (1024*x3)), tmp22, None)
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp46, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/b2/cb2t2n4rg64safrttkku6f7pmgvd6vup5uvatzk3m7rctegdlfml.py
# Topologically Sorted Source Nodes: [x_281], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_281 => add_167, erf_22, mul_160, mul_161, mul_162
# Graph fragment:
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_246, 0.5), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_246, 0.7071067811865476), kwargs = {})
#   %erf_22 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_161,), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_22, 1), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_160, %add_167), kwargs = {})
triton_poi_fused_gelu_16 = async_compile.triton('triton_poi_fused_gelu_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_16(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2129920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 4096
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


# kernel path: /tmp/torchinductor_sahanp/g2/cg2kjyqti2yy3cghbt7nphkyuvcsmffwkh2bwurpnqg6gnswbxau.py
# Topologically Sorted Source Nodes: [x_285, layer_norm_47], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_47 => add_169, add_170, mul_163, mul_164, rsqrt_47, sub_47, var_mean_47
#   x_285 => add_168
# Graph fragment:
#   %add_168 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_164, %view_248), kwargs = {})
#   %var_mean_47 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_168, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_168, %getitem_256), kwargs = {})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_255, 1e-06), kwargs = {})
#   %rsqrt_47 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_169,), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %rsqrt_47), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %arg133_1), kwargs = {})
#   %add_170 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %arg134_1), kwargs = {})
triton_per_fused_add_native_layer_norm_17 = async_compile.triton('triton_per_fused_add_native_layer_norm_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 520
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 1024, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp4 - tmp12
    tmp19 = 1024.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dv/cdvter6enfzmaotafnctdesrrjn4fjxblfdllpxmf6nn3h4bsojn.py
# Topologically Sorted Source Nodes: [x_285, x_290, layer_norm_48], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_48 => add_172, add_173, mul_165, mul_166, rsqrt_48, sub_48, var_mean_48
#   x_285 => add_168
#   x_290 => add_171
# Graph fragment:
#   %add_168 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_164, %view_248), kwargs = {})
#   %add_171 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_168, %view_254), kwargs = {})
#   %var_mean_48 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_171, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_171, %getitem_265), kwargs = {})
#   %add_172 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_264, 1e-06), kwargs = {})
#   %rsqrt_48 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_172,), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %rsqrt_48), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_165, %arg139_1), kwargs = {})
#   %add_173 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_166, %arg140_1), kwargs = {})
triton_per_fused_add_native_layer_norm_18 = async_compile.triton('triton_per_fused_add_native_layer_norm_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 520
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), None)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tl.full([1], 1024, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp9 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tmp8 - tmp16
    tmp23 = 1024.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, None)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/b5/cb5qacfg47u3qrjeqbimhuqq5bnckzk3ldtipx24np3lclgkzim3.py
# Topologically Sorted Source Nodes: [cls_tokens_9], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   cls_tokens_9 => add_190, add_191, clone_82, mul_184, mul_185, rsqrt_53, sub_53, var_mean_53
# Graph fragment:
#   %clone_82 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%slice_23,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_53 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_82, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_82, %getitem_289), kwargs = {})
#   %add_190 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_288, 1e-06), kwargs = {})
#   %rsqrt_53 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_190,), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %rsqrt_53), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %arg169_1), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %arg170_1), kwargs = {})
triton_per_fused_native_layer_norm_19 = async_compile.triton('triton_per_fused_native_layer_norm_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (66560*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (66560*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 1024, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp4 - tmp12
    tmp19 = 1024.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp28, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1 = args
    args.clear()
    assert_size_stride(arg0_1, (256, 3, 14, 14), (588, 196, 14, 1))
    assert_size_stride(arg1_1, (256, ), (1, ))
    assert_size_stride(arg2_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg3_1, (1, 256, 31, 31), (246016, 961, 31, 1))
    assert_size_stride(arg4_1, (1, 1, 256), (256, 256, 1))
    assert_size_stride(arg5_1, (256, ), (1, ))
    assert_size_stride(arg6_1, (256, ), (1, ))
    assert_size_stride(arg7_1, (768, 256), (256, 1))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (256, 256), (256, 1))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (1024, 256), (256, 1))
    assert_size_stride(arg14_1, (1024, ), (1, ))
    assert_size_stride(arg15_1, (256, 1024), (1024, 1))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (768, 256), (256, 1))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (256, 256), (256, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (1024, 256), (256, 1))
    assert_size_stride(arg26_1, (1024, ), (1, ))
    assert_size_stride(arg27_1, (256, 1024), (1024, 1))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (768, 256), (256, 1))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (256, 256), (256, 1))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (1024, 256), (256, 1))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (256, 1024), (1024, 1))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (512, 256), (256, 1))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, ), (1, ))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (1536, 512), (512, 1))
    assert_size_stride(arg48_1, (1536, ), (1, ))
    assert_size_stride(arg49_1, (512, 512), (512, 1))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (2048, 512), (512, 1))
    assert_size_stride(arg54_1, (2048, ), (1, ))
    assert_size_stride(arg55_1, (512, 2048), (2048, 1))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (512, ), (1, ))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (1536, 512), (512, 1))
    assert_size_stride(arg60_1, (1536, ), (1, ))
    assert_size_stride(arg61_1, (512, 512), (512, 1))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (512, ), (1, ))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (2048, 512), (512, 1))
    assert_size_stride(arg66_1, (2048, ), (1, ))
    assert_size_stride(arg67_1, (512, 2048), (2048, 1))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (1536, 512), (512, 1))
    assert_size_stride(arg72_1, (1536, ), (1, ))
    assert_size_stride(arg73_1, (512, 512), (512, 1))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (2048, 512), (512, 1))
    assert_size_stride(arg78_1, (2048, ), (1, ))
    assert_size_stride(arg79_1, (512, 2048), (2048, 1))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (1536, 512), (512, 1))
    assert_size_stride(arg84_1, (1536, ), (1, ))
    assert_size_stride(arg85_1, (512, 512), (512, 1))
    assert_size_stride(arg86_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (2048, 512), (512, 1))
    assert_size_stride(arg90_1, (2048, ), (1, ))
    assert_size_stride(arg91_1, (512, 2048), (2048, 1))
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (512, ), (1, ))
    assert_size_stride(arg95_1, (1536, 512), (512, 1))
    assert_size_stride(arg96_1, (1536, ), (1, ))
    assert_size_stride(arg97_1, (512, 512), (512, 1))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (512, ), (1, ))
    assert_size_stride(arg101_1, (2048, 512), (512, 1))
    assert_size_stride(arg102_1, (2048, ), (1, ))
    assert_size_stride(arg103_1, (512, 2048), (2048, 1))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (512, ), (1, ))
    assert_size_stride(arg107_1, (1536, 512), (512, 1))
    assert_size_stride(arg108_1, (1536, ), (1, ))
    assert_size_stride(arg109_1, (512, 512), (512, 1))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (512, ), (1, ))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (2048, 512), (512, 1))
    assert_size_stride(arg114_1, (2048, ), (1, ))
    assert_size_stride(arg115_1, (512, 2048), (2048, 1))
    assert_size_stride(arg116_1, (512, ), (1, ))
    assert_size_stride(arg117_1, (1024, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, 512), (512, 1))
    assert_size_stride(arg120_1, (1024, ), (1, ))
    assert_size_stride(arg121_1, (1024, ), (1, ))
    assert_size_stride(arg122_1, (1024, ), (1, ))
    assert_size_stride(arg123_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg124_1, (3072, ), (1, ))
    assert_size_stride(arg125_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg126_1, (1024, ), (1, ))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg130_1, (4096, ), (1, ))
    assert_size_stride(arg131_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg136_1, (3072, ), (1, ))
    assert_size_stride(arg137_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg142_1, (4096, ), (1, ))
    assert_size_stride(arg143_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg144_1, (1024, ), (1, ))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg148_1, (3072, ), (1, ))
    assert_size_stride(arg149_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, ), (1, ))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg154_1, (4096, ), (1, ))
    assert_size_stride(arg155_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (1024, ), (1, ))
    assert_size_stride(arg158_1, (1024, ), (1, ))
    assert_size_stride(arg159_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg160_1, (3072, ), (1, ))
    assert_size_stride(arg161_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg166_1, (4096, ), (1, ))
    assert_size_stride(arg167_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg172_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(7, 7), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 256, 31, 31), (246016, 961, 31, 1))
        del arg0_1
        del arg2_1
        buf1 = empty_strided_cuda((8, 962, 1, 2), (1952, 2, 15616, 1), torch.float32)
        buf2 = empty_strided_cuda((8, 962, 1, 2), (1952, 2, 15616, 1), torch.float32)
        buf3 = empty_strided_cuda((8, 962, 1, 2), (1952, 2, 15616, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_165, layer_norm_27], Original ATen: [aten.cat, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_native_layer_norm_0.run(arg4_1, buf0, arg1_1, arg3_1, buf1, buf2, buf3, 15392, 128, grid=grid(15392), stream=stream0)
        buf4 = empty_strided_cuda((8, 962, 1), (962, 1, 7712), torch.float32)
        buf5 = empty_strided_cuda((8, 962, 1), (962, 1, 7712), torch.float32)
        # Topologically Sorted Source Nodes: [x_165, layer_norm_27], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 7696, 2, grid=grid(7696), stream=stream0)
        del buf1
        del buf2
        del buf3
        buf8 = empty_strided_cuda((8, 962, 256), (246272, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_165, layer_norm_27], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_poi_fused_cat_native_layer_norm_2.run(arg4_1, buf0, arg1_1, arg3_1, buf4, buf5, arg5_1, arg6_1, buf8, 1970176, grid=grid(1970176), stream=stream0)
        del arg5_1
        del arg6_1
        del buf4
        del buf5
        buf9 = empty_strided_cuda((7696, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf8, (7696, 256), (256, 1), 0), reinterpret_tensor(arg7_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf9)
        del arg7_1
        del arg8_1
        # Topologically Sorted Source Nodes: [x_166], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf9, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf9, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf9, (8, 4, 962, 64), (738816, 64, 768, 1), 512), None, False)
        buf11 = buf10[0]
        del buf10
        buf15 = reinterpret_tensor(buf8, (7696, 256), (256, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (7696, 256), (256, 1), 0), reinterpret_tensor(arg9_1, (256, 256), (1, 256), 0), out=buf15)
        del arg9_1
        buf16 = reinterpret_tensor(buf15, (8, 962, 256), (246272, 256, 1), 0); del buf15  # reuse
        buf20 = reinterpret_tensor(buf11, (8, 962, 256), (246272, 256, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_165, x_170, layer_norm_28], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_red_fused_add_cat_native_layer_norm_3.run(buf16, arg4_1, buf0, arg1_1, arg3_1, arg10_1, arg11_1, arg12_1, buf20, 7696, 256, grid=grid(7696), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del arg1_1
        del arg3_1
        del arg4_1
        del buf0
        buf21 = empty_strided_cuda((7696, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (7696, 256), (256, 1), 0), reinterpret_tensor(arg13_1, (256, 1024), (1, 256), 0), out=buf21)
        del arg13_1
        buf22 = reinterpret_tensor(buf21, (8, 962, 1024), (985088, 1024, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_172], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf22, arg14_1, 7880704, grid=grid(7880704), stream=stream0)
        del arg14_1
        buf23 = reinterpret_tensor(buf20, (7696, 256), (256, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (7696, 1024), (1024, 1), 0), reinterpret_tensor(arg15_1, (1024, 256), (1, 1024), 0), out=buf23)
        del arg15_1
        buf27 = empty_strided_cuda((8, 962, 256), (246272, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_176, layer_norm_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf16, buf23, arg16_1, arg17_1, arg18_1, buf27, 7696, 256, grid=grid(7696), stream=stream0)
        del arg17_1
        del arg18_1
        buf28 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [linear_59], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg20_1, reinterpret_tensor(buf27, (7696, 256), (256, 1), 0), reinterpret_tensor(arg19_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf28)
        del arg19_1
        del arg20_1
        # Topologically Sorted Source Nodes: [x_177], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf29 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf28, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf28, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf28, (8, 4, 962, 64), (738816, 64, 768, 1), 512), None, False)
        buf30 = buf29[0]
        del buf29
        buf34 = reinterpret_tensor(buf27, (7696, 256), (256, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (7696, 256), (256, 1), 0), reinterpret_tensor(arg21_1, (256, 256), (1, 256), 0), out=buf34)
        del arg21_1
        buf35 = reinterpret_tensor(buf34, (8, 962, 256), (246272, 256, 1), 0); del buf34  # reuse
        buf39 = reinterpret_tensor(buf30, (8, 962, 256), (246272, 256, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_176, x_181, layer_norm_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf35, buf16, buf23, arg16_1, arg22_1, arg23_1, arg24_1, buf39, 7696, 256, grid=grid(7696), stream=stream0)
        del arg16_1
        del arg22_1
        del arg23_1
        del arg24_1
        del buf16
        buf40 = reinterpret_tensor(buf22, (7696, 1024), (1024, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (7696, 256), (256, 1), 0), reinterpret_tensor(arg25_1, (256, 1024), (1, 256), 0), out=buf40)
        del arg25_1
        buf41 = reinterpret_tensor(buf40, (8, 962, 1024), (985088, 1024, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_183], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf41, arg26_1, 7880704, grid=grid(7880704), stream=stream0)
        del arg26_1
        buf42 = reinterpret_tensor(buf39, (7696, 256), (256, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf41, (7696, 1024), (1024, 1), 0), reinterpret_tensor(arg27_1, (1024, 256), (1, 1024), 0), out=buf42)
        del arg27_1
        buf46 = reinterpret_tensor(buf23, (8, 962, 256), (246272, 256, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_187, layer_norm_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf35, buf42, arg28_1, arg29_1, arg30_1, buf46, 7696, 256, grid=grid(7696), stream=stream0)
        del arg29_1
        del arg30_1
        buf47 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg32_1, reinterpret_tensor(buf46, (7696, 256), (256, 1), 0), reinterpret_tensor(arg31_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf47)
        del arg31_1
        del arg32_1
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf48 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf47, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf47, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf47, (8, 4, 962, 64), (738816, 64, 768, 1), 512), None, False)
        del buf47
        buf49 = buf48[0]
        del buf48
        buf53 = reinterpret_tensor(buf46, (7696, 256), (256, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (7696, 256), (256, 1), 0), reinterpret_tensor(arg33_1, (256, 256), (1, 256), 0), out=buf53)
        del arg33_1
        buf54 = reinterpret_tensor(buf53, (8, 962, 256), (246272, 256, 1), 0); del buf53  # reuse
        buf58 = reinterpret_tensor(buf49, (8, 962, 256), (246272, 256, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [x_187, x_192, layer_norm_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf54, buf35, buf42, arg28_1, arg34_1, arg35_1, arg36_1, buf58, 7696, 256, grid=grid(7696), stream=stream0)
        del arg28_1
        del arg34_1
        del arg35_1
        del arg36_1
        del buf35
        del buf42
        buf59 = reinterpret_tensor(buf41, (7696, 1024), (1024, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (7696, 256), (256, 1), 0), reinterpret_tensor(arg37_1, (256, 1024), (1, 256), 0), out=buf59)
        del arg37_1
        buf60 = reinterpret_tensor(buf59, (8, 962, 1024), (985088, 1024, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_194], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf60, arg38_1, 7880704, grid=grid(7880704), stream=stream0)
        del arg38_1
        buf61 = reinterpret_tensor(buf58, (7696, 256), (256, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (7696, 1024), (1024, 1), 0), reinterpret_tensor(arg39_1, (1024, 256), (1, 1024), 0), out=buf61)
        del arg39_1
        del buf60
        buf62 = reinterpret_tensor(buf61, (8, 962, 256), (246272, 256, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_198], Original ATen: [aten.add]
        triton_poi_fused_add_7.run(buf62, buf54, arg40_1, 1970176, grid=grid(1970176), stream=stream0)
        del arg40_1
        del buf54
        buf63 = empty_strided_cuda((8, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cls_token_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (8, 256), (246272, 1), 0), reinterpret_tensor(arg43_1, (256, 512), (1, 256), 0), out=buf63)
        del arg43_1
        # Topologically Sorted Source Nodes: [x_201], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(reinterpret_tensor(buf62, (8, 256, 31, 31), (246272, 1, 7936, 256), 256), arg41_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf64, (8, 512, 16, 16), (131072, 1, 8192, 512))
        del arg41_1
        del buf62
        buf69 = empty_strided_cuda((8, 257, 512), (131584, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_203, layer_norm_33], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_8.run(buf63, arg44_1, buf64, arg42_1, arg45_1, arg46_1, buf69, 2056, 512, grid=grid(2056), stream=stream0)
        del arg45_1
        del arg46_1
        buf70 = empty_strided_cuda((2056, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg48_1, reinterpret_tensor(buf69, (2056, 512), (512, 1), 0), reinterpret_tensor(arg47_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf70)
        del arg47_1
        del arg48_1
        # Topologically Sorted Source Nodes: [x_204], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf71 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf70, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf70, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf70, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, False)
        buf72 = buf71[0]
        del buf71
        buf76 = reinterpret_tensor(buf69, (2056, 512), (512, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (2056, 512), (512, 1), 0), reinterpret_tensor(arg49_1, (512, 512), (1, 512), 0), out=buf76)
        del arg49_1
        buf77 = reinterpret_tensor(buf76, (8, 257, 512), (131584, 512, 1), 0); del buf76  # reuse
        buf81 = reinterpret_tensor(buf72, (8, 257, 512), (131584, 512, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_203, x_208, layer_norm_34], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_9.run(buf77, buf63, arg44_1, buf64, arg42_1, arg50_1, arg51_1, arg52_1, buf81, 2056, 512, grid=grid(2056), stream=stream0)
        del arg42_1
        del arg44_1
        del arg50_1
        del arg51_1
        del arg52_1
        del buf63
        del buf64
        buf82 = empty_strided_cuda((2056, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf81, (2056, 512), (512, 1), 0), reinterpret_tensor(arg53_1, (512, 2048), (1, 512), 0), out=buf82)
        del arg53_1
        buf83 = reinterpret_tensor(buf82, (8, 257, 2048), (526336, 2048, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [x_210], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf83, arg54_1, 4210688, grid=grid(4210688), stream=stream0)
        del arg54_1
        buf84 = reinterpret_tensor(buf81, (2056, 512), (512, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf83, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg55_1, (2048, 512), (1, 2048), 0), out=buf84)
        del arg55_1
        buf88 = empty_strided_cuda((8, 257, 512), (131584, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_214, layer_norm_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf77, buf84, arg56_1, arg57_1, arg58_1, buf88, 2056, 512, grid=grid(2056), stream=stream0)
        del arg57_1
        del arg58_1
        buf89 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg60_1, reinterpret_tensor(buf88, (2056, 512), (512, 1), 0), reinterpret_tensor(arg59_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf89)
        del arg59_1
        del arg60_1
        # Topologically Sorted Source Nodes: [x_215], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf90 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf89, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf89, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf89, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, False)
        buf91 = buf90[0]
        del buf90
        buf95 = reinterpret_tensor(buf88, (2056, 512), (512, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf91, (2056, 512), (512, 1), 0), reinterpret_tensor(arg61_1, (512, 512), (1, 512), 0), out=buf95)
        del arg61_1
        buf96 = reinterpret_tensor(buf95, (8, 257, 512), (131584, 512, 1), 0); del buf95  # reuse
        buf100 = reinterpret_tensor(buf91, (8, 257, 512), (131584, 512, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_214, x_219, layer_norm_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf96, buf77, buf84, arg56_1, arg62_1, arg63_1, arg64_1, buf100, 2056, 512, grid=grid(2056), stream=stream0)
        del arg56_1
        del arg62_1
        del arg63_1
        del arg64_1
        del buf77
        buf101 = reinterpret_tensor(buf83, (2056, 2048), (2048, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (2056, 512), (512, 1), 0), reinterpret_tensor(arg65_1, (512, 2048), (1, 512), 0), out=buf101)
        del arg65_1
        buf102 = reinterpret_tensor(buf101, (8, 257, 2048), (526336, 2048, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf102, arg66_1, 4210688, grid=grid(4210688), stream=stream0)
        del arg66_1
        buf103 = reinterpret_tensor(buf100, (2056, 512), (512, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf102, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg67_1, (2048, 512), (1, 2048), 0), out=buf103)
        del arg67_1
        buf107 = reinterpret_tensor(buf84, (8, 257, 512), (131584, 512, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_225, layer_norm_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf96, buf103, arg68_1, arg69_1, arg70_1, buf107, 2056, 512, grid=grid(2056), stream=stream0)
        del arg69_1
        del arg70_1
        buf108 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [linear_76], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg72_1, reinterpret_tensor(buf107, (2056, 512), (512, 1), 0), reinterpret_tensor(arg71_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf108)
        del arg71_1
        del arg72_1
        # Topologically Sorted Source Nodes: [x_226], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf109 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf108, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf108, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf108, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, False)
        buf110 = buf109[0]
        del buf109
        buf114 = reinterpret_tensor(buf107, (2056, 512), (512, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (2056, 512), (512, 1), 0), reinterpret_tensor(arg73_1, (512, 512), (1, 512), 0), out=buf114)
        del arg73_1
        buf115 = reinterpret_tensor(buf114, (8, 257, 512), (131584, 512, 1), 0); del buf114  # reuse
        buf119 = reinterpret_tensor(buf110, (8, 257, 512), (131584, 512, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_225, x_230, layer_norm_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf115, buf96, buf103, arg68_1, arg74_1, arg75_1, arg76_1, buf119, 2056, 512, grid=grid(2056), stream=stream0)
        del arg68_1
        del arg74_1
        del arg75_1
        del arg76_1
        del buf103
        buf120 = reinterpret_tensor(buf102, (2056, 2048), (2048, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (2056, 512), (512, 1), 0), reinterpret_tensor(arg77_1, (512, 2048), (1, 512), 0), out=buf120)
        del arg77_1
        buf121 = reinterpret_tensor(buf120, (8, 257, 2048), (526336, 2048, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [x_232], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf121, arg78_1, 4210688, grid=grid(4210688), stream=stream0)
        del arg78_1
        buf122 = reinterpret_tensor(buf119, (2056, 512), (512, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg79_1, (2048, 512), (1, 2048), 0), out=buf122)
        del arg79_1
        buf126 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_236, layer_norm_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf115, buf122, arg80_1, arg81_1, arg82_1, buf126, 2056, 512, grid=grid(2056), stream=stream0)
        del arg81_1
        del arg82_1
        buf127 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [linear_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg84_1, reinterpret_tensor(buf126, (2056, 512), (512, 1), 0), reinterpret_tensor(arg83_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf127)
        del arg83_1
        del arg84_1
        # Topologically Sorted Source Nodes: [x_237], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf128 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf127, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf127, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf127, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, False)
        buf129 = buf128[0]
        del buf128
        buf133 = reinterpret_tensor(buf126, (2056, 512), (512, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (2056, 512), (512, 1), 0), reinterpret_tensor(arg85_1, (512, 512), (1, 512), 0), out=buf133)
        del arg85_1
        buf134 = reinterpret_tensor(buf133, (8, 257, 512), (131584, 512, 1), 0); del buf133  # reuse
        buf138 = reinterpret_tensor(buf129, (8, 257, 512), (131584, 512, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [x_236, x_241, layer_norm_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf134, buf115, buf122, arg80_1, arg86_1, arg87_1, arg88_1, buf138, 2056, 512, grid=grid(2056), stream=stream0)
        del arg80_1
        del arg86_1
        del arg87_1
        del arg88_1
        del buf115
        buf139 = reinterpret_tensor(buf121, (2056, 2048), (2048, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (2056, 512), (512, 1), 0), reinterpret_tensor(arg89_1, (512, 2048), (1, 512), 0), out=buf139)
        del arg89_1
        buf140 = reinterpret_tensor(buf139, (8, 257, 2048), (526336, 2048, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_243], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf140, arg90_1, 4210688, grid=grid(4210688), stream=stream0)
        del arg90_1
        buf141 = reinterpret_tensor(buf138, (2056, 512), (512, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg91_1, (2048, 512), (1, 2048), 0), out=buf141)
        del arg91_1
        buf145 = reinterpret_tensor(buf122, (8, 257, 512), (131584, 512, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_247, layer_norm_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf134, buf141, arg92_1, arg93_1, arg94_1, buf145, 2056, 512, grid=grid(2056), stream=stream0)
        del arg93_1
        del arg94_1
        buf146 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [linear_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg96_1, reinterpret_tensor(buf145, (2056, 512), (512, 1), 0), reinterpret_tensor(arg95_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf146)
        del arg95_1
        del arg96_1
        # Topologically Sorted Source Nodes: [x_248], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf147 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf146, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf146, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf146, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, False)
        buf148 = buf147[0]
        del buf147
        buf152 = reinterpret_tensor(buf145, (2056, 512), (512, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (2056, 512), (512, 1), 0), reinterpret_tensor(arg97_1, (512, 512), (1, 512), 0), out=buf152)
        del arg97_1
        buf153 = reinterpret_tensor(buf152, (8, 257, 512), (131584, 512, 1), 0); del buf152  # reuse
        buf157 = reinterpret_tensor(buf148, (8, 257, 512), (131584, 512, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_247, x_252, layer_norm_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf153, buf134, buf141, arg92_1, arg98_1, arg99_1, arg100_1, buf157, 2056, 512, grid=grid(2056), stream=stream0)
        del arg100_1
        del arg92_1
        del arg98_1
        del arg99_1
        del buf134
        buf158 = reinterpret_tensor(buf140, (2056, 2048), (2048, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (2056, 512), (512, 1), 0), reinterpret_tensor(arg101_1, (512, 2048), (1, 512), 0), out=buf158)
        del arg101_1
        buf159 = reinterpret_tensor(buf158, (8, 257, 2048), (526336, 2048, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [x_254], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf159, arg102_1, 4210688, grid=grid(4210688), stream=stream0)
        del arg102_1
        buf160 = reinterpret_tensor(buf157, (2056, 512), (512, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg103_1, (2048, 512), (1, 2048), 0), out=buf160)
        del arg103_1
        buf164 = reinterpret_tensor(buf141, (8, 257, 512), (131584, 512, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [x_258, layer_norm_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf153, buf160, arg104_1, arg105_1, arg106_1, buf164, 2056, 512, grid=grid(2056), stream=stream0)
        del arg105_1
        del arg106_1
        buf165 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg108_1, reinterpret_tensor(buf164, (2056, 512), (512, 1), 0), reinterpret_tensor(arg107_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf165)
        del arg107_1
        del arg108_1
        # Topologically Sorted Source Nodes: [x_259], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf166 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf165, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf165, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf165, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, False)
        del buf165
        buf167 = buf166[0]
        del buf166
        buf171 = reinterpret_tensor(buf164, (2056, 512), (512, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (2056, 512), (512, 1), 0), reinterpret_tensor(arg109_1, (512, 512), (1, 512), 0), out=buf171)
        del arg109_1
        buf172 = reinterpret_tensor(buf171, (8, 257, 512), (131584, 512, 1), 0); del buf171  # reuse
        buf176 = reinterpret_tensor(buf167, (8, 257, 512), (131584, 512, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_258, x_263, layer_norm_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf172, buf153, buf160, arg104_1, arg110_1, arg111_1, arg112_1, buf176, 2056, 512, grid=grid(2056), stream=stream0)
        del arg104_1
        del arg110_1
        del arg111_1
        del arg112_1
        del buf153
        del buf160
        buf177 = reinterpret_tensor(buf159, (2056, 2048), (2048, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (2056, 512), (512, 1), 0), reinterpret_tensor(arg113_1, (512, 2048), (1, 512), 0), out=buf177)
        del arg113_1
        buf178 = reinterpret_tensor(buf177, (8, 257, 2048), (526336, 2048, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_265], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf178, arg114_1, 4210688, grid=grid(4210688), stream=stream0)
        del arg114_1
        buf179 = reinterpret_tensor(buf176, (2056, 512), (512, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg115_1, (2048, 512), (1, 2048), 0), out=buf179)
        del arg115_1
        del buf178
        buf180 = reinterpret_tensor(buf179, (8, 257, 512), (131584, 512, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_269], Original ATen: [aten.add]
        triton_poi_fused_add_13.run(buf180, buf172, arg116_1, 1052672, grid=grid(1052672), stream=stream0)
        del arg116_1
        del buf172
        buf181 = empty_strided_cuda((8, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cls_token_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (8, 512), (131584, 1), 0), reinterpret_tensor(arg119_1, (512, 1024), (1, 512), 0), out=buf181)
        del arg119_1
        # Topologically Sorted Source Nodes: [x_272], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(reinterpret_tensor(buf180, (8, 512, 16, 16), (131584, 1, 8192, 512), 512), arg117_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf182, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
        del arg117_1
        del buf180
        buf187 = empty_strided_cuda((8, 65, 1024), (66560, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_274, layer_norm_45], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_14.run(buf181, arg120_1, buf182, arg118_1, arg121_1, arg122_1, buf187, 520, 1024, grid=grid(520), stream=stream0)
        del arg121_1
        del arg122_1
        buf188 = empty_strided_cuda((520, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg124_1, reinterpret_tensor(buf187, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg123_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf188)
        del arg123_1
        del arg124_1
        # Topologically Sorted Source Nodes: [x_275], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf189 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf188, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf188, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf188, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), None, False)
        buf190 = buf189[0]
        del buf189
        buf194 = reinterpret_tensor(buf187, (520, 1024), (1024, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg125_1, (1024, 1024), (1, 1024), 0), out=buf194)
        del arg125_1
        buf195 = reinterpret_tensor(buf194, (8, 65, 1024), (66560, 1024, 1), 0); del buf194  # reuse
        buf199 = reinterpret_tensor(buf190, (8, 65, 1024), (66560, 1024, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_274, x_279, layer_norm_46], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_15.run(buf195, buf181, arg120_1, buf182, arg118_1, arg126_1, arg127_1, arg128_1, buf199, 520, 1024, grid=grid(520), stream=stream0)
        del arg118_1
        del arg120_1
        del arg126_1
        del arg127_1
        del arg128_1
        del buf182
        buf200 = empty_strided_cuda((520, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg129_1, (1024, 4096), (1, 1024), 0), out=buf200)
        del arg129_1
        buf201 = reinterpret_tensor(buf200, (8, 65, 4096), (266240, 4096, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [x_281], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf201, arg130_1, 2129920, grid=grid(2129920), stream=stream0)
        del arg130_1
        buf202 = reinterpret_tensor(buf199, (520, 1024), (1024, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (520, 4096), (4096, 1), 0), reinterpret_tensor(arg131_1, (4096, 1024), (1, 4096), 0), out=buf202)
        del arg131_1
        buf206 = empty_strided_cuda((8, 65, 1024), (66560, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_285, layer_norm_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf195, buf202, arg132_1, arg133_1, arg134_1, buf206, 520, 1024, grid=grid(520), stream=stream0)
        del arg133_1
        del arg134_1
        buf207 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [linear_97], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg136_1, reinterpret_tensor(buf206, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg135_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf207)
        del arg135_1
        del arg136_1
        # Topologically Sorted Source Nodes: [x_286], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf208 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf207, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf207, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf207, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), None, False)
        buf209 = buf208[0]
        del buf208
        buf213 = reinterpret_tensor(buf206, (520, 1024), (1024, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf209, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 1024), (1, 1024), 0), out=buf213)
        del arg137_1
        buf214 = reinterpret_tensor(buf213, (8, 65, 1024), (66560, 1024, 1), 0); del buf213  # reuse
        buf218 = reinterpret_tensor(buf209, (8, 65, 1024), (66560, 1024, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [x_285, x_290, layer_norm_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_18.run(buf214, buf195, buf202, arg132_1, arg138_1, arg139_1, arg140_1, buf218, 520, 1024, grid=grid(520), stream=stream0)
        del arg132_1
        del arg138_1
        del arg139_1
        del arg140_1
        del buf195
        buf219 = reinterpret_tensor(buf201, (520, 4096), (4096, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg141_1, (1024, 4096), (1, 1024), 0), out=buf219)
        del arg141_1
        buf220 = reinterpret_tensor(buf219, (8, 65, 4096), (266240, 4096, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_292], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf220, arg142_1, 2129920, grid=grid(2129920), stream=stream0)
        del arg142_1
        buf221 = reinterpret_tensor(buf218, (520, 1024), (1024, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (520, 4096), (4096, 1), 0), reinterpret_tensor(arg143_1, (4096, 1024), (1, 4096), 0), out=buf221)
        del arg143_1
        buf225 = reinterpret_tensor(buf202, (8, 65, 1024), (66560, 1024, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [x_296, layer_norm_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf214, buf221, arg144_1, arg145_1, arg146_1, buf225, 520, 1024, grid=grid(520), stream=stream0)
        del arg145_1
        del arg146_1
        buf226 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [linear_101], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg148_1, reinterpret_tensor(buf225, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg147_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf226)
        del arg147_1
        del arg148_1
        # Topologically Sorted Source Nodes: [x_297], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf227 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf226, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf226, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf226, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), None, False)
        buf228 = buf227[0]
        del buf227
        buf232 = reinterpret_tensor(buf225, (520, 1024), (1024, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg149_1, (1024, 1024), (1, 1024), 0), out=buf232)
        del arg149_1
        buf233 = reinterpret_tensor(buf232, (8, 65, 1024), (66560, 1024, 1), 0); del buf232  # reuse
        buf237 = reinterpret_tensor(buf228, (8, 65, 1024), (66560, 1024, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [x_296, x_301, layer_norm_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_18.run(buf233, buf214, buf221, arg144_1, arg150_1, arg151_1, arg152_1, buf237, 520, 1024, grid=grid(520), stream=stream0)
        del arg144_1
        del arg150_1
        del arg151_1
        del arg152_1
        del buf214
        buf238 = reinterpret_tensor(buf220, (520, 4096), (4096, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf237, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg153_1, (1024, 4096), (1, 1024), 0), out=buf238)
        del arg153_1
        buf239 = reinterpret_tensor(buf238, (8, 65, 4096), (266240, 4096, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [x_303], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf239, arg154_1, 2129920, grid=grid(2129920), stream=stream0)
        del arg154_1
        buf240 = reinterpret_tensor(buf237, (520, 1024), (1024, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (520, 4096), (4096, 1), 0), reinterpret_tensor(arg155_1, (4096, 1024), (1, 4096), 0), out=buf240)
        del arg155_1
        buf244 = reinterpret_tensor(buf221, (8, 65, 1024), (66560, 1024, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [x_307, layer_norm_51], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf233, buf240, arg156_1, arg157_1, arg158_1, buf244, 520, 1024, grid=grid(520), stream=stream0)
        del arg157_1
        del arg158_1
        buf245 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [linear_105], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg160_1, reinterpret_tensor(buf244, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg159_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf245)
        del arg159_1
        del arg160_1
        # Topologically Sorted Source Nodes: [x_308], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf246 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf245, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf245, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf245, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), None, False)
        del buf245
        buf247 = buf246[0]
        del buf246
        buf251 = reinterpret_tensor(buf244, (520, 1024), (1024, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg161_1, (1024, 1024), (1, 1024), 0), out=buf251)
        del arg161_1
        buf252 = reinterpret_tensor(buf251, (8, 65, 1024), (66560, 1024, 1), 0); del buf251  # reuse
        buf256 = reinterpret_tensor(buf247, (8, 65, 1024), (66560, 1024, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [x_307, x_312, layer_norm_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_18.run(buf252, buf233, buf240, arg156_1, arg162_1, arg163_1, arg164_1, buf256, 520, 1024, grid=grid(520), stream=stream0)
        del arg156_1
        del arg162_1
        del arg163_1
        del arg164_1
        del buf233
        del buf240
        buf257 = reinterpret_tensor(buf239, (520, 4096), (4096, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg165_1, (1024, 4096), (1, 1024), 0), out=buf257)
        del arg165_1
        buf258 = reinterpret_tensor(buf257, (8, 65, 4096), (266240, 4096, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [x_314], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf258, arg166_1, 2129920, grid=grid(2129920), stream=stream0)
        del arg166_1
        buf259 = reinterpret_tensor(buf256, (520, 1024), (1024, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (520, 4096), (4096, 1), 0), reinterpret_tensor(arg167_1, (4096, 1024), (1, 4096), 0), out=buf259)
        del arg167_1
        del buf258
        buf263 = reinterpret_tensor(buf181, (8, 1, 1024), (1024, 1024, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [cls_tokens_9], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_19.run(buf252, buf259, arg168_1, arg169_1, arg170_1, buf263, 8, 1024, grid=grid(8), stream=stream0)
        del arg168_1
        del arg169_1
        del arg170_1
        del buf252
        del buf259
        buf264 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_323], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg172_1, reinterpret_tensor(buf263, (8, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf264)
        del arg171_1
        del arg172_1
        del buf263
    return (buf264, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256, 3, 14, 14), (588, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 256, 31, 31), (246016, 961, 31, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pit_b_224', benchmark_compiled_module)
