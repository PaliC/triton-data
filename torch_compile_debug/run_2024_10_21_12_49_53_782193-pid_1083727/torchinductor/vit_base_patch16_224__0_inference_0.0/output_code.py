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


# kernel path: /tmp/torchinductor_sahanp/ij/cijwvspnhrmlurx2je6s6nij7mxa3fepsytzh2ky5enr5ygtkz6h.py
# Topologically Sorted Source Nodes: [x_143, x_144, layer_norm_25], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_25 => var_mean_25
#   x_143 => cat_1
#   x_144 => add_87
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %permute_74], 1), kwargs = {})
#   %add_87 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_1, %arg3_1), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_87, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_cat_native_layer_norm_0 = async_compile.triton('triton_red_fused_add_cat_native_layer_norm_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9456
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 6) % 197
    x0 = xindex % 6
    x2 = (xindex // 1182)
    x5 = xindex % 1182
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp15 = tl.load(in_ptr3 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r3 + (128*x0)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 197, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((196*r3) + (25088*x0) + (150528*x2) + (((-1) + x1) % 196)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (r3 + (128*x0)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tl.where(tmp4, tmp5, tmp13)
        tmp16 = tmp14 + tmp15
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
    tl.store(out_ptr0 + (x5 + (1184*x2)), tmp18, xmask)
    tl.store(out_ptr1 + (x5 + (1184*x2)), tmp19, xmask)
    tl.store(out_ptr2 + (x5 + (1184*x2)), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/no/cno5xkodjb2mfkhak5y7w6urgoxswa7essoicimjeli7ery7a26v.py
# Topologically Sorted Source Nodes: [x_143, x_144, layer_norm_25], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_25 => var_mean_25
#   x_143 => cat_1
#   x_144 => add_87
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %permute_74], 1), kwargs = {})
#   %add_87 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_1, %arg3_1), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_87, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_cat_native_layer_norm_1 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1576
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 197
    x1 = (xindex // 197)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (6*x0) + (1184*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (6*x0) + (1184*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2 + (6*x0) + (1184*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oc/cocikbtyoyqvnrwjie5iyorz652jg46td3dxq4n2edfs7kcawu7k.py
# Topologically Sorted Source Nodes: [x_143, x_144, layer_norm_25], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_25 => add_88, add_89, mul_86, mul_87, rsqrt_25, sub_25, var_mean_25
#   x_143 => cat_1
#   x_144 => add_87
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %permute_74], 1), kwargs = {})
#   %add_87 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_1, %arg3_1), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_87, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_87, %getitem_135), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_134, 1e-06), kwargs = {})
#   %rsqrt_25 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_88,), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt_25), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_86, %arg5_1), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %arg6_1), kwargs = {})
triton_poi_fused_add_cat_native_layer_norm_2 = async_compile.triton('triton_poi_fused_add_cat_native_layer_norm_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768) % 197
    x0 = xindex % 768
    x2 = (xindex // 151296)
    x4 = xindex % 151296
    x3 = (xindex // 768)
    x5 = xindex
    tmp15 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((196*x0) + (150528*x2) + (((-1) + x1) % 196)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp5, tmp13)
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 - tmp17
    tmp20 = 768.0
    tmp21 = tmp19 / tmp20
    tmp22 = 1e-06
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp18 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr1 + (x5), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6c/c6cbl6pj5xy3yw5ly224u7q6emjlzd7mbvopjnjyrv372g27ymrk.py
# Topologically Sorted Source Nodes: [x_143, x_144, x_150, layer_norm_26], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_26 => add_91, add_92, mul_88, mul_89, rsqrt_26, sub_26, var_mean_26
#   x_143 => cat_1
#   x_144 => add_87
#   x_150 => add_90
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_1, %permute_74], 1), kwargs = {})
#   %add_87 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_1, %arg3_1), kwargs = {})
#   %add_90 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_87, %view_127), kwargs = {})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_90, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_90, %getitem_144), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_143, 1e-06), kwargs = {})
#   %rsqrt_26 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_91,), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %rsqrt_26), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %arg11_1), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %arg12_1), kwargs = {})
triton_red_fused_add_cat_native_layer_norm_3 = async_compile.triton('triton_red_fused_add_cat_native_layer_norm_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_layer_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1576
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 197
    x1 = (xindex // 197)
    x3 = xindex
    tmp22_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr3 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 197, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((196*r2) + (150528*x1) + (((-1) + x0) % 196)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tl.where(tmp4, tmp5, tmp13)
        tmp16 = tmp14 + tmp15
        tmp19 = tmp17 + tmp18
        tmp20 = tmp16 + tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp22_mean_next, tmp22_m2_next, tmp22_weight_next = triton_helpers.welford_reduce(
            tmp21, tmp22_mean, tmp22_m2, tmp22_weight, roffset == 0
        )
        tmp22_mean = tl.where(rmask & xmask, tmp22_mean_next, tmp22_mean)
        tmp22_m2 = tl.where(rmask & xmask, tmp22_m2_next, tmp22_m2)
        tmp22_weight = tl.where(rmask & xmask, tmp22_weight_next, tmp22_weight)
        tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp20, rmask & xmask)
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
        tmp25 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tmp25 - tmp22
        tmp27 = 768.0
        tmp28 = tmp23 / tmp27
        tmp29 = 1e-06
        tmp30 = tmp28 + tmp29
        tmp31 = libdevice.rsqrt(tmp30)
        tmp32 = tmp26 * tmp31
        tmp34 = tmp32 * tmp33
        tmp36 = tmp34 + tmp35
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp36, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yy/cyyoxk4xyr6jhajqwei6jbebkqwx5jf3s7zqn5pnx5or4mfjaxn2.py
# Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_152 => add_93, erf_12, mul_90, mul_91, mul_92
# Graph fragment:
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_129, 0.5), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_129, 0.7071067811865476), kwargs = {})
#   %erf_12 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_91,), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_12, 1), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %add_93), kwargs = {})
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
    xnumel = 4841472
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


# kernel path: /tmp/torchinductor_sahanp/zh/czhkgzpipxo4rak4yk2q4qm3c7gdzzxljqey7mgfdnt4ytshrkjp.py
# Topologically Sorted Source Nodes: [x_156, layer_norm_27], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_27 => add_95, add_96, mul_93, mul_94, rsqrt_27, sub_27, var_mean_27
#   x_156 => add_94
# Graph fragment:
#   %add_94 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_90, %view_131), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_94, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_94, %getitem_146), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_145, 1e-06), kwargs = {})
#   %rsqrt_27 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_95,), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_27), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, %arg17_1), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %arg18_1), kwargs = {})
triton_per_fused_add_native_layer_norm_5 = async_compile.triton('triton_per_fused_add_native_layer_norm_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1576
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7j/c7ju5ve2r3bm4brdnpz7uj2psu6qfc5vquvr5bj2exm2zdych26x.py
# Topologically Sorted Source Nodes: [x_156, x_161, layer_norm_28], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_28 => add_98, add_99, mul_95, mul_96, rsqrt_28, sub_28, var_mean_28
#   x_156 => add_94
#   x_161 => add_97
# Graph fragment:
#   %add_94 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_90, %view_131), kwargs = {})
#   %add_97 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, %view_137), kwargs = {})
#   %var_mean_28 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_97, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_97, %getitem_155), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_154, 1e-06), kwargs = {})
#   %rsqrt_28 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_98,), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_28), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_95, %arg23_1), kwargs = {})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_96, %arg24_1), kwargs = {})
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_per_fused_add_native_layer_norm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1576
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rv/crv5ejoazmxt3qbybjflcm36cyvzlx3rdqqjqa2ma63o3i2iv7n4.py
# Topologically Sorted Source Nodes: [x_277, x_278], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_277 => add_171
#   x_278 => var_mean_49
# Graph fragment:
#   %add_171 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_167, %view_241), kwargs = {})
#   %var_mean_49 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_171, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_7 = async_compile.triton('triton_per_fused_add_native_layer_norm_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1576
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tl.store(out_ptr0 + (x0), tmp14, None)
    tl.store(out_ptr1 + (x0), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zx/czxkaqgymtavo6h6cavf3hgdbnsvp37ck2ruk6vxcwvjbgk5alhi.py
# Topologically Sorted Source Nodes: [x_280], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_280 => clone_75
# Graph fragment:
#   %clone_75 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%select_1,), kwargs = {})
triton_poi_fused_clone_8 = async_compile.triton('triton_poi_fused_clone_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (151296*x1)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (151296*x1)), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (197*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (197*x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 768.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (1, 197, 768), (151296, 768, 1))
    assert_size_stride(arg4_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (2304, 768), (768, 1))
    assert_size_stride(arg8_1, (2304, ), (1, ))
    assert_size_stride(arg9_1, (768, 768), (768, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (3072, 768), (768, 1))
    assert_size_stride(arg14_1, (3072, ), (1, ))
    assert_size_stride(arg15_1, (768, 3072), (3072, 1))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (2304, 768), (768, 1))
    assert_size_stride(arg20_1, (2304, ), (1, ))
    assert_size_stride(arg21_1, (768, 768), (768, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (3072, 768), (768, 1))
    assert_size_stride(arg26_1, (3072, ), (1, ))
    assert_size_stride(arg27_1, (768, 3072), (3072, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (2304, 768), (768, 1))
    assert_size_stride(arg32_1, (2304, ), (1, ))
    assert_size_stride(arg33_1, (768, 768), (768, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (3072, 768), (768, 1))
    assert_size_stride(arg38_1, (3072, ), (1, ))
    assert_size_stride(arg39_1, (768, 3072), (3072, 1))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (2304, 768), (768, 1))
    assert_size_stride(arg44_1, (2304, ), (1, ))
    assert_size_stride(arg45_1, (768, 768), (768, 1))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (3072, 768), (768, 1))
    assert_size_stride(arg50_1, (3072, ), (1, ))
    assert_size_stride(arg51_1, (768, 3072), (3072, 1))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (2304, 768), (768, 1))
    assert_size_stride(arg56_1, (2304, ), (1, ))
    assert_size_stride(arg57_1, (768, 768), (768, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (3072, 768), (768, 1))
    assert_size_stride(arg62_1, (3072, ), (1, ))
    assert_size_stride(arg63_1, (768, 3072), (3072, 1))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (2304, 768), (768, 1))
    assert_size_stride(arg68_1, (2304, ), (1, ))
    assert_size_stride(arg69_1, (768, 768), (768, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (3072, 768), (768, 1))
    assert_size_stride(arg74_1, (3072, ), (1, ))
    assert_size_stride(arg75_1, (768, 3072), (3072, 1))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (2304, 768), (768, 1))
    assert_size_stride(arg80_1, (2304, ), (1, ))
    assert_size_stride(arg81_1, (768, 768), (768, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (3072, 768), (768, 1))
    assert_size_stride(arg86_1, (3072, ), (1, ))
    assert_size_stride(arg87_1, (768, 3072), (3072, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (2304, 768), (768, 1))
    assert_size_stride(arg92_1, (2304, ), (1, ))
    assert_size_stride(arg93_1, (768, 768), (768, 1))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (3072, 768), (768, 1))
    assert_size_stride(arg98_1, (3072, ), (1, ))
    assert_size_stride(arg99_1, (768, 3072), (3072, 1))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (2304, 768), (768, 1))
    assert_size_stride(arg104_1, (2304, ), (1, ))
    assert_size_stride(arg105_1, (768, 768), (768, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (3072, 768), (768, 1))
    assert_size_stride(arg110_1, (3072, ), (1, ))
    assert_size_stride(arg111_1, (768, 3072), (3072, 1))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (2304, 768), (768, 1))
    assert_size_stride(arg116_1, (2304, ), (1, ))
    assert_size_stride(arg117_1, (768, 768), (768, 1))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (3072, 768), (768, 1))
    assert_size_stride(arg122_1, (3072, ), (1, ))
    assert_size_stride(arg123_1, (768, 3072), (3072, 1))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (2304, 768), (768, 1))
    assert_size_stride(arg128_1, (2304, ), (1, ))
    assert_size_stride(arg129_1, (768, 768), (768, 1))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (3072, 768), (768, 1))
    assert_size_stride(arg134_1, (3072, ), (1, ))
    assert_size_stride(arg135_1, (768, 3072), (3072, 1))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (2304, 768), (768, 1))
    assert_size_stride(arg140_1, (2304, ), (1, ))
    assert_size_stride(arg141_1, (768, 768), (768, 1))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (3072, 768), (768, 1))
    assert_size_stride(arg146_1, (3072, ), (1, ))
    assert_size_stride(arg147_1, (768, 3072), (3072, 1))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (1000, 768), (768, 1))
    assert_size_stride(arg152_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((8, 197, 1, 6), (1184, 6, 9472, 1), torch.float32)
        buf2 = empty_strided_cuda((8, 197, 1, 6), (1184, 6, 9472, 1), torch.float32)
        buf3 = empty_strided_cuda((8, 197, 1, 6), (1184, 6, 9472, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_143, x_144, layer_norm_25], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_native_layer_norm_0.run(arg4_1, buf0, arg2_1, arg3_1, buf1, buf2, buf3, 9456, 128, grid=grid(9456), stream=stream0)
        buf4 = empty_strided_cuda((8, 197, 1), (197, 1, 1600), torch.float32)
        buf5 = empty_strided_cuda((8, 197, 1), (197, 1, 1600), torch.float32)
        # Topologically Sorted Source Nodes: [x_143, x_144, layer_norm_25], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 1576, 6, grid=grid(1576), stream=stream0)
        del buf1
        del buf2
        del buf3
        buf8 = empty_strided_cuda((8, 197, 768), (151296, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_143, x_144, layer_norm_25], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_poi_fused_add_cat_native_layer_norm_2.run(arg4_1, buf0, arg2_1, arg3_1, buf4, buf5, arg5_1, arg6_1, buf8, 1210368, grid=grid(1210368), stream=stream0)
        del arg5_1
        del arg6_1
        buf9 = empty_strided_cuda((1576, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg8_1, reinterpret_tensor(buf8, (1576, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf9)
        del arg7_1
        del arg8_1
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf9, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf9, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf9, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf11 = buf10[0]
        del buf10
        buf15 = reinterpret_tensor(buf8, (1576, 768), (768, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (1576, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), out=buf15)
        del arg9_1
        buf16 = reinterpret_tensor(buf15, (8, 197, 768), (151296, 768, 1), 0); del buf15  # reuse
        buf20 = reinterpret_tensor(buf11, (8, 197, 768), (151296, 768, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_143, x_144, x_150, layer_norm_26], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_red_fused_add_cat_native_layer_norm_3.run(buf16, arg4_1, buf0, arg2_1, arg3_1, arg10_1, arg11_1, arg12_1, buf20, 1576, 768, grid=grid(1576), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del arg2_1
        del arg3_1
        del arg4_1
        del buf0
        buf21 = empty_strided_cuda((1576, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (1576, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 3072), (1, 768), 0), out=buf21)
        del arg13_1
        buf22 = reinterpret_tensor(buf21, (8, 197, 3072), (605184, 3072, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf22, arg14_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg14_1
        buf23 = reinterpret_tensor(buf20, (1576, 768), (768, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg15_1, (3072, 768), (1, 3072), 0), out=buf23)
        del arg15_1
        buf27 = empty_strided_cuda((8, 197, 768), (151296, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_156, layer_norm_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf16, buf23, arg16_1, arg17_1, arg18_1, buf27, 1576, 768, grid=grid(1576), stream=stream0)
        del arg17_1
        del arg18_1
        buf28 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg20_1, reinterpret_tensor(buf27, (1576, 768), (768, 1), 0), reinterpret_tensor(arg19_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf28)
        del arg19_1
        del arg20_1
        # Topologically Sorted Source Nodes: [x_157], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf29 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf28, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf28, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf28, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf30 = buf29[0]
        del buf29
        buf34 = reinterpret_tensor(buf27, (1576, 768), (768, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (1576, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 768), (1, 768), 0), out=buf34)
        del arg21_1
        buf35 = reinterpret_tensor(buf34, (8, 197, 768), (151296, 768, 1), 0); del buf34  # reuse
        buf39 = reinterpret_tensor(buf30, (8, 197, 768), (151296, 768, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_156, x_161, layer_norm_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf35, buf16, buf23, arg16_1, arg22_1, arg23_1, arg24_1, buf39, 1576, 768, grid=grid(1576), stream=stream0)
        del arg16_1
        del arg22_1
        del arg23_1
        del arg24_1
        del buf16
        buf40 = reinterpret_tensor(buf22, (1576, 3072), (3072, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (1576, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 3072), (1, 768), 0), out=buf40)
        del arg25_1
        buf41 = reinterpret_tensor(buf40, (8, 197, 3072), (605184, 3072, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf41, arg26_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg26_1
        buf42 = reinterpret_tensor(buf39, (1576, 768), (768, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf41, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg27_1, (3072, 768), (1, 3072), 0), out=buf42)
        del arg27_1
        buf46 = reinterpret_tensor(buf23, (8, 197, 768), (151296, 768, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_167, layer_norm_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf35, buf42, arg28_1, arg29_1, arg30_1, buf46, 1576, 768, grid=grid(1576), stream=stream0)
        del arg29_1
        del arg30_1
        buf47 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg32_1, reinterpret_tensor(buf46, (1576, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf47)
        del arg31_1
        del arg32_1
        # Topologically Sorted Source Nodes: [x_168], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf48 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf47, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf47, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf47, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf49 = buf48[0]
        del buf48
        buf53 = reinterpret_tensor(buf46, (1576, 768), (768, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (1576, 768), (768, 1), 0), reinterpret_tensor(arg33_1, (768, 768), (1, 768), 0), out=buf53)
        del arg33_1
        buf54 = reinterpret_tensor(buf53, (8, 197, 768), (151296, 768, 1), 0); del buf53  # reuse
        buf58 = reinterpret_tensor(buf49, (8, 197, 768), (151296, 768, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [x_167, x_172, layer_norm_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf54, buf35, buf42, arg28_1, arg34_1, arg35_1, arg36_1, buf58, 1576, 768, grid=grid(1576), stream=stream0)
        del arg28_1
        del arg34_1
        del arg35_1
        del arg36_1
        del buf35
        buf59 = reinterpret_tensor(buf41, (1576, 3072), (3072, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (1576, 768), (768, 1), 0), reinterpret_tensor(arg37_1, (768, 3072), (1, 768), 0), out=buf59)
        del arg37_1
        buf60 = reinterpret_tensor(buf59, (8, 197, 3072), (605184, 3072, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_174], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf60, arg38_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg38_1
        buf61 = reinterpret_tensor(buf58, (1576, 768), (768, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg39_1, (3072, 768), (1, 3072), 0), out=buf61)
        del arg39_1
        buf65 = reinterpret_tensor(buf42, (8, 197, 768), (151296, 768, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_178, layer_norm_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf54, buf61, arg40_1, arg41_1, arg42_1, buf65, 1576, 768, grid=grid(1576), stream=stream0)
        del arg41_1
        del arg42_1
        buf66 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg44_1, reinterpret_tensor(buf65, (1576, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf66)
        del arg43_1
        del arg44_1
        # Topologically Sorted Source Nodes: [x_179], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf67 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf66, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf66, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf66, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf68 = buf67[0]
        del buf67
        buf72 = reinterpret_tensor(buf65, (1576, 768), (768, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (1576, 768), (768, 1), 0), reinterpret_tensor(arg45_1, (768, 768), (1, 768), 0), out=buf72)
        del arg45_1
        buf73 = reinterpret_tensor(buf72, (8, 197, 768), (151296, 768, 1), 0); del buf72  # reuse
        buf77 = reinterpret_tensor(buf68, (8, 197, 768), (151296, 768, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_178, x_183, layer_norm_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf73, buf54, buf61, arg40_1, arg46_1, arg47_1, arg48_1, buf77, 1576, 768, grid=grid(1576), stream=stream0)
        del arg40_1
        del arg46_1
        del arg47_1
        del arg48_1
        del buf54
        buf78 = reinterpret_tensor(buf60, (1576, 3072), (3072, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (1576, 768), (768, 1), 0), reinterpret_tensor(arg49_1, (768, 3072), (1, 768), 0), out=buf78)
        del arg49_1
        buf79 = reinterpret_tensor(buf78, (8, 197, 3072), (605184, 3072, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_185], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf79, arg50_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg50_1
        buf80 = reinterpret_tensor(buf77, (1576, 768), (768, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg51_1, (3072, 768), (1, 3072), 0), out=buf80)
        del arg51_1
        buf84 = reinterpret_tensor(buf61, (8, 197, 768), (151296, 768, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_189, layer_norm_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf73, buf80, arg52_1, arg53_1, arg54_1, buf84, 1576, 768, grid=grid(1576), stream=stream0)
        del arg53_1
        del arg54_1
        buf85 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg56_1, reinterpret_tensor(buf84, (1576, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf85)
        del arg55_1
        del arg56_1
        # Topologically Sorted Source Nodes: [x_190], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf86 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf85, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf85, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf85, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf87 = buf86[0]
        del buf86
        buf91 = reinterpret_tensor(buf84, (1576, 768), (768, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (1576, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), out=buf91)
        del arg57_1
        buf92 = reinterpret_tensor(buf91, (8, 197, 768), (151296, 768, 1), 0); del buf91  # reuse
        buf96 = reinterpret_tensor(buf87, (8, 197, 768), (151296, 768, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_189, x_194, layer_norm_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf92, buf73, buf80, arg52_1, arg58_1, arg59_1, arg60_1, buf96, 1576, 768, grid=grid(1576), stream=stream0)
        del arg52_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf73
        buf97 = reinterpret_tensor(buf79, (1576, 3072), (3072, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (1576, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 3072), (1, 768), 0), out=buf97)
        del arg61_1
        buf98 = reinterpret_tensor(buf97, (8, 197, 3072), (605184, 3072, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_196], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf98, arg62_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg62_1
        buf99 = reinterpret_tensor(buf96, (1576, 768), (768, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg63_1, (3072, 768), (1, 3072), 0), out=buf99)
        del arg63_1
        buf103 = reinterpret_tensor(buf80, (8, 197, 768), (151296, 768, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_200, layer_norm_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf92, buf99, arg64_1, arg65_1, arg66_1, buf103, 1576, 768, grid=grid(1576), stream=stream0)
        del arg65_1
        del arg66_1
        buf104 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [linear_69], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg68_1, reinterpret_tensor(buf103, (1576, 768), (768, 1), 0), reinterpret_tensor(arg67_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf104)
        del arg67_1
        del arg68_1
        # Topologically Sorted Source Nodes: [x_201], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf105 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf104, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf104, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf104, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf106 = buf105[0]
        del buf105
        buf110 = reinterpret_tensor(buf103, (1576, 768), (768, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (1576, 768), (768, 1), 0), reinterpret_tensor(arg69_1, (768, 768), (1, 768), 0), out=buf110)
        del arg69_1
        buf111 = reinterpret_tensor(buf110, (8, 197, 768), (151296, 768, 1), 0); del buf110  # reuse
        buf115 = reinterpret_tensor(buf106, (8, 197, 768), (151296, 768, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_200, x_205, layer_norm_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf111, buf92, buf99, arg64_1, arg70_1, arg71_1, arg72_1, buf115, 1576, 768, grid=grid(1576), stream=stream0)
        del arg64_1
        del arg70_1
        del arg71_1
        del arg72_1
        del buf92
        buf116 = reinterpret_tensor(buf98, (1576, 3072), (3072, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf115, (1576, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 3072), (1, 768), 0), out=buf116)
        del arg73_1
        buf117 = reinterpret_tensor(buf116, (8, 197, 3072), (605184, 3072, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_207], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf117, arg74_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg74_1
        buf118 = reinterpret_tensor(buf115, (1576, 768), (768, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf117, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg75_1, (3072, 768), (1, 3072), 0), out=buf118)
        del arg75_1
        buf122 = reinterpret_tensor(buf99, (8, 197, 768), (151296, 768, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [x_211, layer_norm_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf111, buf118, arg76_1, arg77_1, arg78_1, buf122, 1576, 768, grid=grid(1576), stream=stream0)
        del arg77_1
        del arg78_1
        buf123 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [linear_73], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg80_1, reinterpret_tensor(buf122, (1576, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf123)
        del arg79_1
        del arg80_1
        # Topologically Sorted Source Nodes: [x_212], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf124 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf123, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf123, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf123, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf125 = buf124[0]
        del buf124
        buf129 = reinterpret_tensor(buf122, (1576, 768), (768, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf125, (1576, 768), (768, 1), 0), reinterpret_tensor(arg81_1, (768, 768), (1, 768), 0), out=buf129)
        del arg81_1
        buf130 = reinterpret_tensor(buf129, (8, 197, 768), (151296, 768, 1), 0); del buf129  # reuse
        buf134 = reinterpret_tensor(buf125, (8, 197, 768), (151296, 768, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [x_211, x_216, layer_norm_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf130, buf111, buf118, arg76_1, arg82_1, arg83_1, arg84_1, buf134, 1576, 768, grid=grid(1576), stream=stream0)
        del arg76_1
        del arg82_1
        del arg83_1
        del arg84_1
        del buf111
        buf135 = reinterpret_tensor(buf117, (1576, 3072), (3072, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf134, (1576, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 3072), (1, 768), 0), out=buf135)
        del arg85_1
        buf136 = reinterpret_tensor(buf135, (8, 197, 3072), (605184, 3072, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [x_218], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf136, arg86_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg86_1
        buf137 = reinterpret_tensor(buf134, (1576, 768), (768, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg87_1, (3072, 768), (1, 3072), 0), out=buf137)
        del arg87_1
        buf141 = reinterpret_tensor(buf118, (8, 197, 768), (151296, 768, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_222, layer_norm_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf130, buf137, arg88_1, arg89_1, arg90_1, buf141, 1576, 768, grid=grid(1576), stream=stream0)
        del arg89_1
        del arg90_1
        buf142 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg92_1, reinterpret_tensor(buf141, (1576, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf142)
        del arg91_1
        del arg92_1
        # Topologically Sorted Source Nodes: [x_223], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf143 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf142, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf142, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf142, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf144 = buf143[0]
        del buf143
        buf148 = reinterpret_tensor(buf141, (1576, 768), (768, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf144, (1576, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 768), (1, 768), 0), out=buf148)
        del arg93_1
        buf149 = reinterpret_tensor(buf148, (8, 197, 768), (151296, 768, 1), 0); del buf148  # reuse
        buf153 = reinterpret_tensor(buf144, (8, 197, 768), (151296, 768, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_222, x_227, layer_norm_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf149, buf130, buf137, arg88_1, arg94_1, arg95_1, arg96_1, buf153, 1576, 768, grid=grid(1576), stream=stream0)
        del arg88_1
        del arg94_1
        del arg95_1
        del arg96_1
        del buf130
        buf154 = reinterpret_tensor(buf136, (1576, 3072), (3072, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (1576, 768), (768, 1), 0), reinterpret_tensor(arg97_1, (768, 3072), (1, 768), 0), out=buf154)
        del arg97_1
        buf155 = reinterpret_tensor(buf154, (8, 197, 3072), (605184, 3072, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [x_229], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf155, arg98_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg98_1
        buf156 = reinterpret_tensor(buf153, (1576, 768), (768, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf155, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg99_1, (3072, 768), (1, 3072), 0), out=buf156)
        del arg99_1
        buf160 = reinterpret_tensor(buf137, (8, 197, 768), (151296, 768, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [x_233, layer_norm_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf149, buf156, arg100_1, arg101_1, arg102_1, buf160, 1576, 768, grid=grid(1576), stream=stream0)
        del arg101_1
        del arg102_1
        buf161 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [linear_81], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg104_1, reinterpret_tensor(buf160, (1576, 768), (768, 1), 0), reinterpret_tensor(arg103_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf161)
        del arg103_1
        del arg104_1
        # Topologically Sorted Source Nodes: [x_234], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf162 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf161, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf161, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf161, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf163 = buf162[0]
        del buf162
        buf167 = reinterpret_tensor(buf160, (1576, 768), (768, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf163, (1576, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), out=buf167)
        del arg105_1
        buf168 = reinterpret_tensor(buf167, (8, 197, 768), (151296, 768, 1), 0); del buf167  # reuse
        buf172 = reinterpret_tensor(buf163, (8, 197, 768), (151296, 768, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_233, x_238, layer_norm_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf168, buf149, buf156, arg100_1, arg106_1, arg107_1, arg108_1, buf172, 1576, 768, grid=grid(1576), stream=stream0)
        del arg100_1
        del arg106_1
        del arg107_1
        del arg108_1
        del buf149
        buf173 = reinterpret_tensor(buf155, (1576, 3072), (3072, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (1576, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 3072), (1, 768), 0), out=buf173)
        del arg109_1
        buf174 = reinterpret_tensor(buf173, (8, 197, 3072), (605184, 3072, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [x_240], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf174, arg110_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg110_1
        buf175 = reinterpret_tensor(buf172, (1576, 768), (768, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg111_1, (3072, 768), (1, 3072), 0), out=buf175)
        del arg111_1
        buf179 = reinterpret_tensor(buf156, (8, 197, 768), (151296, 768, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [x_244, layer_norm_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf168, buf175, arg112_1, arg113_1, arg114_1, buf179, 1576, 768, grid=grid(1576), stream=stream0)
        del arg113_1
        del arg114_1
        buf180 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg116_1, reinterpret_tensor(buf179, (1576, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf180)
        del arg115_1
        del arg116_1
        # Topologically Sorted Source Nodes: [x_245], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf181 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf180, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf180, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf180, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf182 = buf181[0]
        del buf181
        buf186 = reinterpret_tensor(buf179, (1576, 768), (768, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1576, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 768), (1, 768), 0), out=buf186)
        del arg117_1
        buf187 = reinterpret_tensor(buf186, (8, 197, 768), (151296, 768, 1), 0); del buf186  # reuse
        buf191 = reinterpret_tensor(buf182, (8, 197, 768), (151296, 768, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [x_244, x_249, layer_norm_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf187, buf168, buf175, arg112_1, arg118_1, arg119_1, arg120_1, buf191, 1576, 768, grid=grid(1576), stream=stream0)
        del arg112_1
        del arg118_1
        del arg119_1
        del arg120_1
        del buf168
        buf192 = reinterpret_tensor(buf174, (1576, 3072), (3072, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (1576, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 3072), (1, 768), 0), out=buf192)
        del arg121_1
        buf193 = reinterpret_tensor(buf192, (8, 197, 3072), (605184, 3072, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [x_251], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf193, arg122_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg122_1
        buf194 = reinterpret_tensor(buf191, (1576, 768), (768, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf193, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg123_1, (3072, 768), (1, 3072), 0), out=buf194)
        del arg123_1
        buf198 = reinterpret_tensor(buf175, (8, 197, 768), (151296, 768, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [x_255, layer_norm_45], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf187, buf194, arg124_1, arg125_1, arg126_1, buf198, 1576, 768, grid=grid(1576), stream=stream0)
        del arg125_1
        del arg126_1
        buf199 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg128_1, reinterpret_tensor(buf198, (1576, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf199)
        del arg127_1
        del arg128_1
        # Topologically Sorted Source Nodes: [x_256], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf200 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf199, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf199, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf199, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        buf201 = buf200[0]
        del buf200
        buf205 = reinterpret_tensor(buf198, (1576, 768), (768, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (1576, 768), (768, 1), 0), reinterpret_tensor(arg129_1, (768, 768), (1, 768), 0), out=buf205)
        del arg129_1
        buf206 = reinterpret_tensor(buf205, (8, 197, 768), (151296, 768, 1), 0); del buf205  # reuse
        buf210 = reinterpret_tensor(buf201, (8, 197, 768), (151296, 768, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [x_255, x_260, layer_norm_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf206, buf187, buf194, arg124_1, arg130_1, arg131_1, arg132_1, buf210, 1576, 768, grid=grid(1576), stream=stream0)
        del arg124_1
        del arg130_1
        del arg131_1
        del arg132_1
        del buf187
        buf211 = reinterpret_tensor(buf193, (1576, 3072), (3072, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (1576, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 3072), (1, 768), 0), out=buf211)
        del arg133_1
        buf212 = reinterpret_tensor(buf211, (8, 197, 3072), (605184, 3072, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [x_262], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf212, arg134_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg134_1
        buf213 = reinterpret_tensor(buf210, (1576, 768), (768, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg135_1, (3072, 768), (1, 3072), 0), out=buf213)
        del arg135_1
        buf217 = reinterpret_tensor(buf194, (8, 197, 768), (151296, 768, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [x_266, layer_norm_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_5.run(buf206, buf213, arg136_1, arg137_1, arg138_1, buf217, 1576, 768, grid=grid(1576), stream=stream0)
        del arg137_1
        del arg138_1
        buf218 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg140_1, reinterpret_tensor(buf217, (1576, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf218)
        del arg139_1
        del arg140_1
        # Topologically Sorted Source Nodes: [x_267], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf219 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf218, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf218, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf218, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, False)
        del buf218
        buf220 = buf219[0]
        del buf219
        buf224 = reinterpret_tensor(buf217, (1576, 768), (768, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (1576, 768), (768, 1), 0), reinterpret_tensor(arg141_1, (768, 768), (1, 768), 0), out=buf224)
        del arg141_1
        buf225 = reinterpret_tensor(buf224, (8, 197, 768), (151296, 768, 1), 0); del buf224  # reuse
        buf229 = reinterpret_tensor(buf220, (8, 197, 768), (151296, 768, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [x_266, x_271, layer_norm_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf225, buf206, buf213, arg136_1, arg142_1, arg143_1, arg144_1, buf229, 1576, 768, grid=grid(1576), stream=stream0)
        del arg136_1
        del arg142_1
        del arg143_1
        del arg144_1
        del buf206
        del buf213
        buf230 = reinterpret_tensor(buf212, (1576, 3072), (3072, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (1576, 768), (768, 1), 0), reinterpret_tensor(arg145_1, (768, 3072), (1, 768), 0), out=buf230)
        del arg145_1
        buf231 = reinterpret_tensor(buf230, (8, 197, 3072), (605184, 3072, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [x_273], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf231, arg146_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg146_1
        buf232 = reinterpret_tensor(buf229, (1576, 768), (768, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg147_1, (3072, 768), (1, 3072), 0), out=buf232)
        del arg147_1
        del buf231
        buf233 = buf5; del buf5  # reuse
        buf234 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_277, x_278], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf225, buf232, arg148_1, buf233, buf234, 1576, 768, grid=grid(1576), stream=stream0)
        buf236 = empty_strided_cuda((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_280], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf225, buf232, arg148_1, buf233, buf234, arg149_1, arg150_1, buf236, 6144, grid=grid(6144), stream=stream0)
        del arg148_1
        del arg149_1
        del arg150_1
        del buf225
        del buf232
        del buf233
        del buf234
        buf237 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_280, x_281], Original ATen: [aten.clone, aten.addmm]
        extern_kernels.addmm(arg152_1, buf236, reinterpret_tensor(arg151_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf237)
        del arg151_1
        del arg152_1
        del buf236
    return (buf237, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('vit_base_patch16_224', benchmark_compiled_module)
