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


# kernel path: /tmp/torchinductor_sahanp/5l/c5lc3d64w56mjjf7wqs6jidnyc5zbbbyaoyzarkr7r6k6nmjm4gl.py
# Topologically Sorted Source Nodes: [x_143, x_144, layer_norm_25], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_25 => add_89, add_90, mul_86, mul_87, rsqrt_25, sub_25, var_mean_25
#   x_143 => cat_1
#   x_144 => add_88
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_2, %expand_3, %permute_75], 1), kwargs = {})
#   %add_88 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_1, %arg3_1), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_88, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_88, %getitem_135), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_134, 1e-06), kwargs = {})
#   %rsqrt_25 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_89,), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt_25), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_86, %arg6_1), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %arg7_1), kwargs = {})
triton_red_fused_add_cat_native_layer_norm_0 = async_compile.triton('triton_red_fused_add_cat_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1584
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 198
    x1 = (xindex // 198)
    x3 = xindex
    tmp24_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp21 = tl.load(in_ptr4 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 2, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tmp6 & tmp8
        tmp10 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp0 >= tmp7
        tmp12 = tl.full([1, 1], 198, tl.int64)
        tmp13 = tmp0 < tmp12
        tmp14 = tl.load(in_ptr2 + ((196*r2) + (150528*x1) + (((-2) + x0) % 196)), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp14 + tmp15
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp11, tmp16, tmp17)
        tmp19 = tl.where(tmp9, tmp10, tmp18)
        tmp20 = tl.where(tmp4, tmp5, tmp19)
        tmp22 = tmp20 + tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp24_mean_next, tmp24_m2_next, tmp24_weight_next = triton_helpers.welford_reduce(
            tmp23, tmp24_mean, tmp24_m2, tmp24_weight, roffset == 0
        )
        tmp24_mean = tl.where(rmask & xmask, tmp24_mean_next, tmp24_mean)
        tmp24_m2 = tl.where(rmask & xmask, tmp24_m2_next, tmp24_m2)
        tmp24_weight = tl.where(rmask & xmask, tmp24_weight_next, tmp24_weight)
        tl.store(out_ptr0 + (r2 + (768*x3)), tmp22, rmask & xmask)
    tmp24_tmp, tmp25_tmp, tmp26_tmp = triton_helpers.welford(
        tmp24_mean, tmp24_m2, tmp24_weight, 1
    )
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp27 = tl.load(out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tmp27 - tmp24
        tmp29 = 768.0
        tmp30 = tmp25 / tmp29
        tmp31 = 1e-06
        tmp32 = tmp30 + tmp31
        tmp33 = libdevice.rsqrt(tmp32)
        tmp34 = tmp28 * tmp33
        tmp36 = tmp34 * tmp35
        tmp38 = tmp36 + tmp37
        tl.store(out_ptr3 + (r2 + (768*x3)), tmp38, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rm/crmjquwt5s4ldxzwrlcwa6re6a4fd33jf3ddjc7keeztiv3pgjcc.py
# Topologically Sorted Source Nodes: [x_150, layer_norm_26], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_26 => add_92, add_93, mul_88, mul_89, rsqrt_26, sub_26, var_mean_26
#   x_150 => add_91
# Graph fragment:
#   %add_91 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %view_127), kwargs = {})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_91, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_91, %getitem_144), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_143, 1e-06), kwargs = {})
#   %rsqrt_26 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_92,), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %rsqrt_26), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %arg12_1), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %arg13_1), kwargs = {})
triton_per_fused_add_native_layer_norm_1 = async_compile.triton('triton_per_fused_add_native_layer_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1584
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


# kernel path: /tmp/torchinductor_sahanp/az/cazqxr3hbobs3r2qdqx5yonjnxztr75igyhbf5piod4uxr3tvuwl.py
# Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_152 => add_94, erf_12, mul_90, mul_91, mul_92
# Graph fragment:
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_129, 0.5), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_129, 0.7071067811865476), kwargs = {})
#   %erf_12 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_91,), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_12, 1), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %add_94), kwargs = {})
triton_poi_fused_gelu_2 = async_compile.triton('triton_poi_fused_gelu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4866048
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


# kernel path: /tmp/torchinductor_sahanp/aa/caaqlyfx5oeomeqs36zaves46plf3p2hkv6v6k4mc6xg57262yjb.py
# Topologically Sorted Source Nodes: [x_150, x_156, layer_norm_27], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_27 => add_96, add_97, mul_93, mul_94, rsqrt_27, sub_27, var_mean_27
#   x_150 => add_91
#   x_156 => add_95
# Graph fragment:
#   %add_91 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %view_127), kwargs = {})
#   %add_95 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, %view_131), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_95, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_95, %getitem_146), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_145, 1e-06), kwargs = {})
#   %rsqrt_27 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_96,), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_27), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, %arg18_1), kwargs = {})
#   %add_97 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %arg19_1), kwargs = {})
triton_per_fused_add_native_layer_norm_3 = async_compile.triton('triton_per_fused_add_native_layer_norm_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1584
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


# kernel path: /tmp/torchinductor_sahanp/ha/chajxv2y24wpa5blknyv4hmutyfzffyhoxtfex5wmwrjfkylwhhh.py
# Topologically Sorted Source Nodes: [add_51, x_281], Original ATen: [aten.add, aten.div]
# Source node to ATen node mapping:
#   add_51 => add_175
#   x_281 => div_1
# Graph fragment:
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %arg153_1), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg155_1), kwargs = {})
#   %add_175 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_1, %add_tensor), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_175, 2), kwargs = {})
triton_poi_fused_add_div_4 = async_compile.triton('triton_poi_fused_add_div_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1000
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (1, 198, 768), (152064, 768, 1))
    assert_size_stride(arg4_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg5_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (2304, 768), (768, 1))
    assert_size_stride(arg9_1, (2304, ), (1, ))
    assert_size_stride(arg10_1, (768, 768), (768, 1))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (3072, 768), (768, 1))
    assert_size_stride(arg15_1, (3072, ), (1, ))
    assert_size_stride(arg16_1, (768, 3072), (3072, 1))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (2304, 768), (768, 1))
    assert_size_stride(arg21_1, (2304, ), (1, ))
    assert_size_stride(arg22_1, (768, 768), (768, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (3072, 768), (768, 1))
    assert_size_stride(arg27_1, (3072, ), (1, ))
    assert_size_stride(arg28_1, (768, 3072), (3072, 1))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (2304, 768), (768, 1))
    assert_size_stride(arg33_1, (2304, ), (1, ))
    assert_size_stride(arg34_1, (768, 768), (768, 1))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (3072, 768), (768, 1))
    assert_size_stride(arg39_1, (3072, ), (1, ))
    assert_size_stride(arg40_1, (768, 3072), (3072, 1))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (2304, 768), (768, 1))
    assert_size_stride(arg45_1, (2304, ), (1, ))
    assert_size_stride(arg46_1, (768, 768), (768, 1))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (3072, 768), (768, 1))
    assert_size_stride(arg51_1, (3072, ), (1, ))
    assert_size_stride(arg52_1, (768, 3072), (3072, 1))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (2304, 768), (768, 1))
    assert_size_stride(arg57_1, (2304, ), (1, ))
    assert_size_stride(arg58_1, (768, 768), (768, 1))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (3072, 768), (768, 1))
    assert_size_stride(arg63_1, (3072, ), (1, ))
    assert_size_stride(arg64_1, (768, 3072), (3072, 1))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (2304, 768), (768, 1))
    assert_size_stride(arg69_1, (2304, ), (1, ))
    assert_size_stride(arg70_1, (768, 768), (768, 1))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (3072, 768), (768, 1))
    assert_size_stride(arg75_1, (3072, ), (1, ))
    assert_size_stride(arg76_1, (768, 3072), (3072, 1))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (2304, 768), (768, 1))
    assert_size_stride(arg81_1, (2304, ), (1, ))
    assert_size_stride(arg82_1, (768, 768), (768, 1))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (3072, 768), (768, 1))
    assert_size_stride(arg87_1, (3072, ), (1, ))
    assert_size_stride(arg88_1, (768, 3072), (3072, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (2304, 768), (768, 1))
    assert_size_stride(arg93_1, (2304, ), (1, ))
    assert_size_stride(arg94_1, (768, 768), (768, 1))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (3072, 768), (768, 1))
    assert_size_stride(arg99_1, (3072, ), (1, ))
    assert_size_stride(arg100_1, (768, 3072), (3072, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (2304, 768), (768, 1))
    assert_size_stride(arg105_1, (2304, ), (1, ))
    assert_size_stride(arg106_1, (768, 768), (768, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (3072, 768), (768, 1))
    assert_size_stride(arg111_1, (3072, ), (1, ))
    assert_size_stride(arg112_1, (768, 3072), (3072, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (2304, 768), (768, 1))
    assert_size_stride(arg117_1, (2304, ), (1, ))
    assert_size_stride(arg118_1, (768, 768), (768, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (3072, 768), (768, 1))
    assert_size_stride(arg123_1, (3072, ), (1, ))
    assert_size_stride(arg124_1, (768, 3072), (3072, 1))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (2304, 768), (768, 1))
    assert_size_stride(arg129_1, (2304, ), (1, ))
    assert_size_stride(arg130_1, (768, 768), (768, 1))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (3072, 768), (768, 1))
    assert_size_stride(arg135_1, (3072, ), (1, ))
    assert_size_stride(arg136_1, (768, 3072), (3072, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (2304, 768), (768, 1))
    assert_size_stride(arg141_1, (2304, ), (1, ))
    assert_size_stride(arg142_1, (768, 768), (768, 1))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (3072, 768), (768, 1))
    assert_size_stride(arg147_1, (3072, ), (1, ))
    assert_size_stride(arg148_1, (768, 3072), (3072, 1))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (1000, 768), (768, 1))
    assert_size_stride(arg153_1, (1000, ), (1, ))
    assert_size_stride(arg154_1, (1000, 768), (768, 1))
    assert_size_stride(arg155_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((8, 198, 768), (152064, 768, 1), torch.float32)
        buf5 = empty_strided_cuda((8, 198, 768), (152064, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_143, x_144, layer_norm_25], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_native_layer_norm_0.run(arg4_1, arg5_1, buf0, arg2_1, arg3_1, arg6_1, arg7_1, buf1, buf5, 1584, 768, grid=grid(1584), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del buf0
        buf6 = empty_strided_cuda((1584, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg9_1, reinterpret_tensor(buf5, (1584, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf6)
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf6, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf6, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf6, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf8 = buf7[0]
        del buf7
        buf12 = reinterpret_tensor(buf5, (1584, 768), (768, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf8, (1584, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 768), (1, 768), 0), out=buf12)
        del arg10_1
        buf16 = reinterpret_tensor(buf8, (8, 198, 768), (152064, 768, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_150, layer_norm_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf1, buf12, arg11_1, arg12_1, arg13_1, buf16, 1584, 768, grid=grid(1584), stream=stream0)
        del arg12_1
        del arg13_1
        buf17 = empty_strided_cuda((1584, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf16, (1584, 768), (768, 1), 0), reinterpret_tensor(arg14_1, (768, 3072), (1, 768), 0), out=buf17)
        del arg14_1
        buf18 = reinterpret_tensor(buf17, (8, 198, 3072), (608256, 3072, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf18, arg15_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg15_1
        buf19 = reinterpret_tensor(buf16, (1584, 768), (768, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg16_1, (3072, 768), (1, 3072), 0), out=buf19)
        del arg16_1
        buf20 = reinterpret_tensor(buf19, (8, 198, 768), (152064, 768, 1), 0); del buf19  # reuse
        buf24 = empty_strided_cuda((8, 198, 768), (152064, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_150, x_156, layer_norm_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf20, buf1, buf12, arg11_1, arg17_1, arg18_1, arg19_1, buf24, 1584, 768, grid=grid(1584), stream=stream0)
        del arg11_1
        del arg17_1
        del arg18_1
        del arg19_1
        del buf1
        buf25 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg21_1, reinterpret_tensor(buf24, (1584, 768), (768, 1), 0), reinterpret_tensor(arg20_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf25)
        del arg20_1
        del arg21_1
        # Topologically Sorted Source Nodes: [x_157], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf26 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf25, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf25, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf25, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf27 = buf26[0]
        del buf26
        buf31 = reinterpret_tensor(buf24, (1584, 768), (768, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (1584, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 768), (1, 768), 0), out=buf31)
        del arg22_1
        buf35 = reinterpret_tensor(buf27, (8, 198, 768), (152064, 768, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [x_161, layer_norm_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf20, buf31, arg23_1, arg24_1, arg25_1, buf35, 1584, 768, grid=grid(1584), stream=stream0)
        del arg24_1
        del arg25_1
        buf36 = reinterpret_tensor(buf18, (1584, 3072), (3072, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf35, (1584, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 3072), (1, 768), 0), out=buf36)
        del arg26_1
        buf37 = reinterpret_tensor(buf36, (8, 198, 3072), (608256, 3072, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf37, arg27_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg27_1
        buf38 = reinterpret_tensor(buf35, (1584, 768), (768, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf37, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg28_1, (3072, 768), (1, 3072), 0), out=buf38)
        del arg28_1
        buf39 = reinterpret_tensor(buf38, (8, 198, 768), (152064, 768, 1), 0); del buf38  # reuse
        buf43 = reinterpret_tensor(buf12, (8, 198, 768), (152064, 768, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_161, x_167, layer_norm_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf39, buf20, buf31, arg23_1, arg29_1, arg30_1, arg31_1, buf43, 1584, 768, grid=grid(1584), stream=stream0)
        del arg23_1
        del arg29_1
        del arg30_1
        del arg31_1
        del buf20
        buf44 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg33_1, reinterpret_tensor(buf43, (1584, 768), (768, 1), 0), reinterpret_tensor(arg32_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf44)
        del arg32_1
        del arg33_1
        # Topologically Sorted Source Nodes: [x_168], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf45 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf44, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf44, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf44, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf46 = buf45[0]
        del buf45
        buf50 = reinterpret_tensor(buf43, (1584, 768), (768, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (1584, 768), (768, 1), 0), reinterpret_tensor(arg34_1, (768, 768), (1, 768), 0), out=buf50)
        del arg34_1
        buf54 = reinterpret_tensor(buf46, (8, 198, 768), (152064, 768, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_172, layer_norm_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf39, buf50, arg35_1, arg36_1, arg37_1, buf54, 1584, 768, grid=grid(1584), stream=stream0)
        del arg36_1
        del arg37_1
        buf55 = reinterpret_tensor(buf37, (1584, 3072), (3072, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf54, (1584, 768), (768, 1), 0), reinterpret_tensor(arg38_1, (768, 3072), (1, 768), 0), out=buf55)
        del arg38_1
        buf56 = reinterpret_tensor(buf55, (8, 198, 3072), (608256, 3072, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_174], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf56, arg39_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg39_1
        buf57 = reinterpret_tensor(buf54, (1584, 768), (768, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg40_1, (3072, 768), (1, 3072), 0), out=buf57)
        del arg40_1
        buf58 = reinterpret_tensor(buf57, (8, 198, 768), (152064, 768, 1), 0); del buf57  # reuse
        buf62 = reinterpret_tensor(buf31, (8, 198, 768), (152064, 768, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_172, x_178, layer_norm_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf58, buf39, buf50, arg35_1, arg41_1, arg42_1, arg43_1, buf62, 1584, 768, grid=grid(1584), stream=stream0)
        del arg35_1
        del arg41_1
        del arg42_1
        del arg43_1
        del buf39
        buf63 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg45_1, reinterpret_tensor(buf62, (1584, 768), (768, 1), 0), reinterpret_tensor(arg44_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf63)
        del arg44_1
        del arg45_1
        # Topologically Sorted Source Nodes: [x_179], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf64 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf63, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf63, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf63, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf65 = buf64[0]
        del buf64
        buf69 = reinterpret_tensor(buf62, (1584, 768), (768, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (1584, 768), (768, 1), 0), reinterpret_tensor(arg46_1, (768, 768), (1, 768), 0), out=buf69)
        del arg46_1
        buf73 = reinterpret_tensor(buf65, (8, 198, 768), (152064, 768, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_183, layer_norm_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf58, buf69, arg47_1, arg48_1, arg49_1, buf73, 1584, 768, grid=grid(1584), stream=stream0)
        del arg48_1
        del arg49_1
        buf74 = reinterpret_tensor(buf56, (1584, 3072), (3072, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (1584, 768), (768, 1), 0), reinterpret_tensor(arg50_1, (768, 3072), (1, 768), 0), out=buf74)
        del arg50_1
        buf75 = reinterpret_tensor(buf74, (8, 198, 3072), (608256, 3072, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_185], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf75, arg51_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg51_1
        buf76 = reinterpret_tensor(buf73, (1584, 768), (768, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg52_1, (3072, 768), (1, 3072), 0), out=buf76)
        del arg52_1
        buf77 = reinterpret_tensor(buf76, (8, 198, 768), (152064, 768, 1), 0); del buf76  # reuse
        buf81 = reinterpret_tensor(buf50, (8, 198, 768), (152064, 768, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_183, x_189, layer_norm_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf77, buf58, buf69, arg47_1, arg53_1, arg54_1, arg55_1, buf81, 1584, 768, grid=grid(1584), stream=stream0)
        del arg47_1
        del arg53_1
        del arg54_1
        del arg55_1
        del buf58
        buf82 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [linear_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg57_1, reinterpret_tensor(buf81, (1584, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf82)
        del arg56_1
        del arg57_1
        # Topologically Sorted Source Nodes: [x_190], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf83 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf82, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf82, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf82, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf84 = buf83[0]
        del buf83
        buf88 = reinterpret_tensor(buf81, (1584, 768), (768, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (1584, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 768), (1, 768), 0), out=buf88)
        del arg58_1
        buf92 = reinterpret_tensor(buf84, (8, 198, 768), (152064, 768, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_194, layer_norm_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf77, buf88, arg59_1, arg60_1, arg61_1, buf92, 1584, 768, grid=grid(1584), stream=stream0)
        del arg60_1
        del arg61_1
        buf93 = reinterpret_tensor(buf75, (1584, 3072), (3072, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (1584, 768), (768, 1), 0), reinterpret_tensor(arg62_1, (768, 3072), (1, 768), 0), out=buf93)
        del arg62_1
        buf94 = reinterpret_tensor(buf93, (8, 198, 3072), (608256, 3072, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_196], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf94, arg63_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg63_1
        buf95 = reinterpret_tensor(buf92, (1584, 768), (768, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg64_1, (3072, 768), (1, 3072), 0), out=buf95)
        del arg64_1
        buf96 = reinterpret_tensor(buf95, (8, 198, 768), (152064, 768, 1), 0); del buf95  # reuse
        buf100 = reinterpret_tensor(buf69, (8, 198, 768), (152064, 768, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_194, x_200, layer_norm_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf96, buf77, buf88, arg59_1, arg65_1, arg66_1, arg67_1, buf100, 1584, 768, grid=grid(1584), stream=stream0)
        del arg59_1
        del arg65_1
        del arg66_1
        del arg67_1
        del buf77
        buf101 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg69_1, reinterpret_tensor(buf100, (1584, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf101)
        del arg68_1
        del arg69_1
        # Topologically Sorted Source Nodes: [x_201], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf102 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf101, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf101, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf101, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf103 = buf102[0]
        del buf102
        buf107 = reinterpret_tensor(buf100, (1584, 768), (768, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (1584, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 768), (1, 768), 0), out=buf107)
        del arg70_1
        buf111 = reinterpret_tensor(buf103, (8, 198, 768), (152064, 768, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [x_205, layer_norm_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf96, buf107, arg71_1, arg72_1, arg73_1, buf111, 1584, 768, grid=grid(1584), stream=stream0)
        del arg72_1
        del arg73_1
        buf112 = reinterpret_tensor(buf94, (1584, 3072), (3072, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (1584, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 3072), (1, 768), 0), out=buf112)
        del arg74_1
        buf113 = reinterpret_tensor(buf112, (8, 198, 3072), (608256, 3072, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_207], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf113, arg75_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg75_1
        buf114 = reinterpret_tensor(buf111, (1584, 768), (768, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf113, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg76_1, (3072, 768), (1, 3072), 0), out=buf114)
        del arg76_1
        buf115 = reinterpret_tensor(buf114, (8, 198, 768), (152064, 768, 1), 0); del buf114  # reuse
        buf119 = reinterpret_tensor(buf88, (8, 198, 768), (152064, 768, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_205, x_211, layer_norm_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf115, buf96, buf107, arg71_1, arg77_1, arg78_1, arg79_1, buf119, 1584, 768, grid=grid(1584), stream=stream0)
        del arg71_1
        del arg77_1
        del arg78_1
        del arg79_1
        del buf107
        buf120 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [linear_74], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg81_1, reinterpret_tensor(buf119, (1584, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf120)
        del arg80_1
        del arg81_1
        # Topologically Sorted Source Nodes: [x_212], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf121 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf120, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf120, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf120, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf122 = buf121[0]
        del buf121
        buf126 = reinterpret_tensor(buf119, (1584, 768), (768, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (1584, 768), (768, 1), 0), reinterpret_tensor(arg82_1, (768, 768), (1, 768), 0), out=buf126)
        del arg82_1
        buf130 = reinterpret_tensor(buf122, (8, 198, 768), (152064, 768, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_216, layer_norm_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf115, buf126, arg83_1, arg84_1, arg85_1, buf130, 1584, 768, grid=grid(1584), stream=stream0)
        del arg84_1
        del arg85_1
        buf131 = reinterpret_tensor(buf113, (1584, 3072), (3072, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (1584, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 3072), (1, 768), 0), out=buf131)
        del arg86_1
        buf132 = reinterpret_tensor(buf131, (8, 198, 3072), (608256, 3072, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_218], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf132, arg87_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg87_1
        buf133 = reinterpret_tensor(buf130, (1584, 768), (768, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf132, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg88_1, (3072, 768), (1, 3072), 0), out=buf133)
        del arg88_1
        buf134 = reinterpret_tensor(buf133, (8, 198, 768), (152064, 768, 1), 0); del buf133  # reuse
        buf138 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_216, x_222, layer_norm_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf134, buf115, buf126, arg83_1, arg89_1, arg90_1, arg91_1, buf138, 1584, 768, grid=grid(1584), stream=stream0)
        del arg83_1
        del arg89_1
        del arg90_1
        del arg91_1
        del buf115
        buf139 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg93_1, reinterpret_tensor(buf138, (1584, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf139)
        del arg92_1
        del arg93_1
        # Topologically Sorted Source Nodes: [x_223], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf140 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf139, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf139, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf139, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf141 = buf140[0]
        del buf140
        buf145 = reinterpret_tensor(buf138, (1584, 768), (768, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (1584, 768), (768, 1), 0), reinterpret_tensor(arg94_1, (768, 768), (1, 768), 0), out=buf145)
        del arg94_1
        buf149 = reinterpret_tensor(buf141, (8, 198, 768), (152064, 768, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [x_227, layer_norm_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf134, buf145, arg95_1, arg96_1, arg97_1, buf149, 1584, 768, grid=grid(1584), stream=stream0)
        del arg96_1
        del arg97_1
        buf150 = reinterpret_tensor(buf132, (1584, 3072), (3072, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (1584, 768), (768, 1), 0), reinterpret_tensor(arg98_1, (768, 3072), (1, 768), 0), out=buf150)
        del arg98_1
        buf151 = reinterpret_tensor(buf150, (8, 198, 3072), (608256, 3072, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [x_229], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf151, arg99_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg99_1
        buf152 = reinterpret_tensor(buf149, (1584, 768), (768, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg100_1, (3072, 768), (1, 3072), 0), out=buf152)
        del arg100_1
        buf153 = reinterpret_tensor(buf152, (8, 198, 768), (152064, 768, 1), 0); del buf152  # reuse
        buf157 = reinterpret_tensor(buf126, (8, 198, 768), (152064, 768, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_227, x_233, layer_norm_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf153, buf134, buf145, arg95_1, arg101_1, arg102_1, arg103_1, buf157, 1584, 768, grid=grid(1584), stream=stream0)
        del arg101_1
        del arg102_1
        del arg103_1
        del arg95_1
        del buf134
        buf158 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg105_1, reinterpret_tensor(buf157, (1584, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf158)
        del arg104_1
        del arg105_1
        # Topologically Sorted Source Nodes: [x_234], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf159 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf158, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf158, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf158, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf160 = buf159[0]
        del buf159
        buf164 = reinterpret_tensor(buf157, (1584, 768), (768, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (1584, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 768), (1, 768), 0), out=buf164)
        del arg106_1
        buf168 = reinterpret_tensor(buf160, (8, 198, 768), (152064, 768, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_238, layer_norm_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf153, buf164, arg107_1, arg108_1, arg109_1, buf168, 1584, 768, grid=grid(1584), stream=stream0)
        del arg108_1
        del arg109_1
        buf169 = reinterpret_tensor(buf151, (1584, 3072), (3072, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf168, (1584, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 3072), (1, 768), 0), out=buf169)
        del arg110_1
        buf170 = reinterpret_tensor(buf169, (8, 198, 3072), (608256, 3072, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_240], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf170, arg111_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg111_1
        buf171 = reinterpret_tensor(buf168, (1584, 768), (768, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg112_1, (3072, 768), (1, 3072), 0), out=buf171)
        del arg112_1
        buf172 = reinterpret_tensor(buf171, (8, 198, 768), (152064, 768, 1), 0); del buf171  # reuse
        buf176 = reinterpret_tensor(buf145, (8, 198, 768), (152064, 768, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [x_238, x_244, layer_norm_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf172, buf153, buf164, arg107_1, arg113_1, arg114_1, arg115_1, buf176, 1584, 768, grid=grid(1584), stream=stream0)
        del arg107_1
        del arg113_1
        del arg114_1
        del arg115_1
        del buf153
        buf177 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg117_1, reinterpret_tensor(buf176, (1584, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf177)
        del arg116_1
        del arg117_1
        # Topologically Sorted Source Nodes: [x_245], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf178 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf177, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf177, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf177, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf179 = buf178[0]
        del buf178
        buf183 = reinterpret_tensor(buf176, (1584, 768), (768, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (1584, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 768), (1, 768), 0), out=buf183)
        del arg118_1
        buf187 = reinterpret_tensor(buf179, (8, 198, 768), (152064, 768, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_249, layer_norm_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf172, buf183, arg119_1, arg120_1, arg121_1, buf187, 1584, 768, grid=grid(1584), stream=stream0)
        del arg120_1
        del arg121_1
        buf188 = reinterpret_tensor(buf170, (1584, 3072), (3072, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (1584, 768), (768, 1), 0), reinterpret_tensor(arg122_1, (768, 3072), (1, 768), 0), out=buf188)
        del arg122_1
        buf189 = reinterpret_tensor(buf188, (8, 198, 3072), (608256, 3072, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_251], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf189, arg123_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg123_1
        buf190 = reinterpret_tensor(buf187, (1584, 768), (768, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg124_1, (3072, 768), (1, 3072), 0), out=buf190)
        del arg124_1
        buf191 = reinterpret_tensor(buf190, (8, 198, 768), (152064, 768, 1), 0); del buf190  # reuse
        buf195 = reinterpret_tensor(buf164, (8, 198, 768), (152064, 768, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [x_249, x_255, layer_norm_45], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf191, buf172, buf183, arg119_1, arg125_1, arg126_1, arg127_1, buf195, 1584, 768, grid=grid(1584), stream=stream0)
        del arg119_1
        del arg125_1
        del arg126_1
        del arg127_1
        del buf172
        buf196 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [linear_90], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg129_1, reinterpret_tensor(buf195, (1584, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf196)
        del arg128_1
        del arg129_1
        # Topologically Sorted Source Nodes: [x_256], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf197 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        buf198 = buf197[0]
        del buf197
        buf202 = reinterpret_tensor(buf195, (1584, 768), (768, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (1584, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 768), (1, 768), 0), out=buf202)
        del arg130_1
        buf206 = reinterpret_tensor(buf198, (8, 198, 768), (152064, 768, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [x_260, layer_norm_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf191, buf202, arg131_1, arg132_1, arg133_1, buf206, 1584, 768, grid=grid(1584), stream=stream0)
        del arg132_1
        del arg133_1
        buf207 = reinterpret_tensor(buf189, (1584, 3072), (3072, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (1584, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 3072), (1, 768), 0), out=buf207)
        del arg134_1
        buf208 = reinterpret_tensor(buf207, (8, 198, 3072), (608256, 3072, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_262], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf208, arg135_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg135_1
        buf209 = reinterpret_tensor(buf206, (1584, 768), (768, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf208, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg136_1, (3072, 768), (1, 3072), 0), out=buf209)
        del arg136_1
        buf210 = reinterpret_tensor(buf209, (8, 198, 768), (152064, 768, 1), 0); del buf209  # reuse
        buf214 = reinterpret_tensor(buf183, (8, 198, 768), (152064, 768, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [x_260, x_266, layer_norm_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf210, buf191, buf202, arg131_1, arg137_1, arg138_1, arg139_1, buf214, 1584, 768, grid=grid(1584), stream=stream0)
        del arg131_1
        del arg137_1
        del arg138_1
        del arg139_1
        del buf191
        buf215 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [linear_94], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg141_1, reinterpret_tensor(buf214, (1584, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf215)
        del arg140_1
        del arg141_1
        # Topologically Sorted Source Nodes: [x_267], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf216 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf215, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf215, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf215, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), None, False)
        del buf215
        buf217 = buf216[0]
        del buf216
        buf221 = reinterpret_tensor(buf214, (1584, 768), (768, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (1584, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 768), (1, 768), 0), out=buf221)
        del arg142_1
        buf225 = reinterpret_tensor(buf217, (8, 198, 768), (152064, 768, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_271, layer_norm_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf210, buf221, arg143_1, arg144_1, arg145_1, buf225, 1584, 768, grid=grid(1584), stream=stream0)
        del arg144_1
        del arg145_1
        buf226 = reinterpret_tensor(buf208, (1584, 3072), (3072, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf225, (1584, 768), (768, 1), 0), reinterpret_tensor(arg146_1, (768, 3072), (1, 768), 0), out=buf226)
        del arg146_1
        buf227 = reinterpret_tensor(buf226, (8, 198, 3072), (608256, 3072, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [x_273], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_2.run(buf227, arg147_1, 4866048, grid=grid(4866048), stream=stream0)
        del arg147_1
        buf228 = reinterpret_tensor(buf225, (1584, 768), (768, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf227, (1584, 3072), (3072, 1), 0), reinterpret_tensor(arg148_1, (3072, 768), (1, 3072), 0), out=buf228)
        del arg148_1
        del buf227
        buf229 = reinterpret_tensor(buf228, (8, 198, 768), (152064, 768, 1), 0); del buf228  # reuse
        buf233 = reinterpret_tensor(buf202, (8, 198, 768), (152064, 768, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [x_271, x_277, x_278], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_3.run(buf229, buf210, buf221, arg143_1, arg149_1, arg150_1, arg151_1, buf233, 1584, 768, grid=grid(1584), stream=stream0)
        del arg143_1
        del arg149_1
        del arg150_1
        del arg151_1
        del buf210
        del buf221
        del buf229
        buf234 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (8, 768), (152064, 1), 0), reinterpret_tensor(arg152_1, (768, 1000), (1, 768), 0), out=buf234)
        del arg152_1
        buf235 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf233, (8, 768), (152064, 1), 768), reinterpret_tensor(arg154_1, (768, 1000), (1, 768), 0), out=buf235)
        del arg154_1
        del buf233
        buf236 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [add_51, x_281], Original ATen: [aten.add, aten.div]
        triton_poi_fused_add_div_4.run(buf236, arg153_1, buf235, arg155_1, 8000, grid=grid(8000), stream=stream0)
        del arg153_1
        del arg155_1
        del buf235
    return (buf236, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('deit_base_distilled_patch16_224', benchmark_compiled_module)
