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


# kernel path: /tmp/torchinductor_sahanp/jw/cjwrhwqkkgz5t2jk6ea52tjfedp47ydb2cjc44phfizsc52lmsih.py
# Topologically Sorted Source Nodes: [layer_norm_61], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_61 => clone_152, var_mean_61
# Graph fragment:
#   %clone_152 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_152,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_61 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_152, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_red_fused_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 2
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r3) + (25088*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/j2/cj246hl5ycrcwdvxyaxbyc2tl3koua3p6benjd7voi2pahqidkds.py
# Topologically Sorted Source Nodes: [layer_norm_61], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_61 => clone_152, var_mean_61
# Graph fragment:
#   %clone_152 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_152,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_61 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_152, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_1 = async_compile.triton('triton_per_fused_native_layer_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (392*x1)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (392*x1)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (196*r2) + (392*x1)), xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/fg/cfgsc4e2hqjbizgnornefyzzwwcfxn25vpxnn5wly6bjqf3kxggx.py
# Topologically Sorted Source Nodes: [layer_norm_61], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_61 => add_212, add_213, clone_152, mul_242, mul_243, rsqrt_61, sub_61, var_mean_61
# Graph fragment:
#   %clone_152 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_152,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_61 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_152, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_152, %getitem_183), kwargs = {})
#   %add_212 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_182, 1e-06), kwargs = {})
#   %rsqrt_61 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_212,), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %rsqrt_61), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_242, %arg3_1), kwargs = {})
#   %add_213 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_243, %arg4_1), kwargs = {})
triton_poi_fused_native_layer_norm_2 = async_compile.triton('triton_poi_fused_native_layer_norm_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zp/czpiqdo5jcveocm6xt3ookgde44st7w6ck74pxhdr7wg73oftpv4.py
# Topologically Sorted Source Nodes: [v_91, v_92], Original ATen: [aten.native_layer_norm, aten.clone]
# Source node to ATen node mapping:
#   v_91 => clone_154, var_mean_62
#   v_92 => clone_155
# Graph fragment:
#   %clone_154 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_185,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_62 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_154, [2]), kwargs = {correction: 0, keepdim: True})
#   %clone_155 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_154,), kwargs = {memory_format: torch.contiguous_format})
triton_red_fused_clone_native_layer_norm_3 = async_compile.triton('triton_red_fused_clone_native_layer_norm_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_layer_norm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_clone_native_layer_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (768 + r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (768 + r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = 0.5
        tmp4 = tmp2 * tmp3
        tmp5 = 0.7071067811865476
        tmp6 = tmp2 * tmp5
        tmp7 = libdevice.erf(tmp6)
        tmp8 = 1.0
        tmp9 = tmp7 + tmp8
        tmp10 = tmp4 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    x2 = xindex % 196
    x3 = (xindex // 196)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr0 + (768 + r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (768 + r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 + tmp16
        tmp18 = 0.5
        tmp19 = tmp17 * tmp18
        tmp20 = 0.7071067811865476
        tmp21 = tmp17 * tmp20
        tmp22 = libdevice.erf(tmp21)
        tmp23 = 1.0
        tmp24 = tmp22 + tmp23
        tmp25 = tmp19 * tmp24
        tmp26 = tmp25 - tmp12
        tmp27 = 768.0
        tmp28 = tmp13 / tmp27
        tmp29 = 1e-05
        tmp30 = tmp28 + tmp29
        tmp31 = libdevice.rsqrt(tmp30)
        tmp32 = tmp26 * tmp31
        tmp34 = tmp32 * tmp33
        tmp36 = tmp34 + tmp35
        tl.store(out_ptr2 + (x2 + (196*r1) + (150528*x3)), tmp36, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7c/c7cdtwlogac7aabdhrlumllglsvfbze2svcurfi3pi3wa3djryvi.py
# Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   x_221 => mul_249
# Graph fragment:
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_184, %permute_156), kwargs = {})
triton_poi_fused_mul_4 = async_compile.triton('triton_poi_fused_mul_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 * tmp13
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ja/cjafzrsrrhbhkufgxo2jjbvdpzlmlelshi5xjutdo7ko46thlyqz.py
# Topologically Sorted Source Nodes: [x_224, layer_norm_63], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_63 => clone_157, var_mean_63
#   x_224 => add_218
# Graph fragment:
#   %add_218 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_152, %view_187), kwargs = {})
#   %clone_157 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_218,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_63 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_157, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_5 = async_compile.triton('triton_red_fused_add_native_layer_norm_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 196
    x2 = (xindex // 392)
    x5 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (50176*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp8, xmask)
    tl.store(out_ptr1 + (x5), tmp9, xmask)
    tl.store(out_ptr2 + (x5), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hu/chuda27xer3r3moos7vdytz5pik3cnr5it2rr4ipnrstqopeg3ns.py
# Topologically Sorted Source Nodes: [x_224, layer_norm_63], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_63 => clone_157, var_mean_63
#   x_224 => add_218
# Graph fragment:
#   %add_218 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_152, %view_187), kwargs = {})
#   %clone_157 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_218,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_63 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_157, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_per_fused_add_native_layer_norm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (2*x0)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (2*x0)), xmask, other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7v/c7v6rnu6yce3mzlkkfaeaan76o2kotioww3unzlsbbigujre5hx2.py
# Topologically Sorted Source Nodes: [x_224, layer_norm_63], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_63 => add_219, add_220, clone_157, mul_250, mul_251, rsqrt_63, sub_63, var_mean_63
#   x_224 => add_218
# Graph fragment:
#   %add_218 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_152, %view_187), kwargs = {})
#   %clone_157 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_218,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_63 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_157, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_157, %getitem_189), kwargs = {})
#   %add_219 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_188, 1e-06), kwargs = {})
#   %rsqrt_63 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_219,), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %rsqrt_63), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_250, %arg13_1), kwargs = {})
#   %add_220 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_251, %arg14_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_7 = async_compile.triton('triton_poi_fused_add_native_layer_norm_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 256.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/un/cun4mmeedmr5tsbbgm3aklclaujhdia4p2thqbngvnlzofafblb4.py
# Topologically Sorted Source Nodes: [x_224, x_231, layer_norm_65], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_65 => add_226, add_227, clone_162, mul_258, mul_259, rsqrt_65, sub_65, var_mean_65
#   x_224 => add_218
#   x_231 => add_225
# Graph fragment:
#   %add_218 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_152, %view_187), kwargs = {})
#   %add_225 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_218, %view_193), kwargs = {})
#   %clone_162 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_225,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_65 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_162, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_162, %getitem_195), kwargs = {})
#   %add_226 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_194, 1e-06), kwargs = {})
#   %rsqrt_65 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_226,), kwargs = {})
#   %mul_258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %rsqrt_65), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_258, %arg23_1), kwargs = {})
#   %add_227 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_259, %arg24_1), kwargs = {})
triton_red_fused_add_native_layer_norm_8 = async_compile.triton('triton_red_fused_add_native_layer_norm_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (50176*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp9 = tmp7 + tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
        tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp10, rmask & xmask)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp15 - tmp12
        tmp17 = 256.0
        tmp18 = tmp13 / tmp17
        tmp19 = 1e-06
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp24 = tmp22 * tmp23
        tmp26 = tmp24 + tmp25
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp26, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bb/cbbew5xhyh3xxxkezx54kc4xo2lq7ahx6tcurw4bnlkqyhmmcyll.py
# Topologically Sorted Source Nodes: [x_238, layer_norm_67], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_67 => add_233, add_234, clone_167, mul_266, mul_267, rsqrt_67, sub_67, var_mean_67
#   x_238 => add_232
# Graph fragment:
#   %add_232 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_225, %view_199), kwargs = {})
#   %clone_167 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_232,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_67 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_167, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_167, %getitem_201), kwargs = {})
#   %add_233 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_200, 1e-06), kwargs = {})
#   %rsqrt_67 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_233,), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %rsqrt_67), kwargs = {})
#   %mul_267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_266, %arg33_1), kwargs = {})
#   %add_234 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_267, %arg34_1), kwargs = {})
triton_per_fused_add_native_layer_norm_9 = async_compile.triton('triton_per_fused_add_native_layer_norm_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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


# kernel path: /tmp/torchinductor_sahanp/jb/cjb4n7jm3lgk3bk2omij5xssmtqocwomxjcso5veit7a75umkh5a.py
# Topologically Sorted Source Nodes: [x_238, x_245, layer_norm_69], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_69 => add_240, add_241, clone_172, mul_274, mul_275, rsqrt_69, sub_69, var_mean_69
#   x_238 => add_232
#   x_245 => add_239
# Graph fragment:
#   %add_232 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_225, %view_199), kwargs = {})
#   %add_239 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_232, %view_205), kwargs = {})
#   %clone_172 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_239,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_69 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_172, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_172, %getitem_207), kwargs = {})
#   %add_240 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_206, 1e-06), kwargs = {})
#   %rsqrt_69 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_240,), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %rsqrt_69), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_274, %arg43_1), kwargs = {})
#   %add_241 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_275, %arg44_1), kwargs = {})
triton_per_fused_add_native_layer_norm_10 = async_compile.triton('triton_per_fused_add_native_layer_norm_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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


# kernel path: /tmp/torchinductor_sahanp/3i/c3ihv2gpd6kmcqrgtk7ro4kghbgidh4q7rpxpam6w4edabghzl4u.py
# Topologically Sorted Source Nodes: [x_420, x_427, x_428], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_420 => add_414
#   x_427 => add_421
#   x_428 => clone_302, var_mean_121
# Graph fragment:
#   %add_414 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_407, %view_355), kwargs = {})
#   %add_421 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_414, %view_361), kwargs = {})
#   %clone_302 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_421,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_121 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_302, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_11 = async_compile.triton('triton_per_fused_add_native_layer_norm_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1568
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
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp8, None)
    tl.store(out_ptr0 + (x0), tmp16, None)
    tl.store(out_ptr1 + (x0), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hw/chwhyrszdyuzvrvyaxwbtxyqeo2zvuth7m4moudnklipsuqpx6np.py
# Topologically Sorted Source Nodes: [x_428, x_429], Original ATen: [aten.native_layer_norm, aten.mean]
# Source node to ATen node mapping:
#   x_428 => add_422, add_423, mul_482, mul_483, rsqrt_121, sub_121, var_mean_121
#   x_429 => mean_1
# Graph fragment:
#   %var_mean_121 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_302, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_121 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_302, %getitem_363), kwargs = {})
#   %add_422 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_362, 1e-06), kwargs = {})
#   %rsqrt_121 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_422,), kwargs = {})
#   %mul_482 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_121, %rsqrt_121), kwargs = {})
#   %mul_483 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_482, %arg303_1), kwargs = {})
#   %add_423 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_483, %arg304_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_423, [1]), kwargs = {})
triton_red_fused_mean_native_layer_norm_12 = async_compile.triton('triton_red_fused_mean_native_layer_norm_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_native_layer_norm_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_native_layer_norm_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (98*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (98*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 256.0
        tmp5 = tmp3 / tmp4
        tmp6 = 1e-06
        tmp7 = tmp5 + tmp6
        tmp8 = libdevice.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6p/c6ptdju6tyenpbt3jzmttngam5mvlw6u427xvi72akstduwpmeir.py
# Topologically Sorted Source Nodes: [x_428, x_429], Original ATen: [aten.native_layer_norm, aten.mean]
# Source node to ATen node mapping:
#   x_428 => add_422, add_423, mul_482, mul_483, rsqrt_121, sub_121, var_mean_121
#   x_429 => mean_1
# Graph fragment:
#   %var_mean_121 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_302, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_121 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_302, %getitem_363), kwargs = {})
#   %add_422 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_362, 1e-06), kwargs = {})
#   %rsqrt_121 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_422,), kwargs = {})
#   %mul_482 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_121, %rsqrt_121), kwargs = {})
#   %mul_483 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_482, %arg303_1), kwargs = {})
#   %add_423 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_483, %arg304_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_423, [1]), kwargs = {})
triton_per_fused_mean_native_layer_norm_13 = async_compile.triton('triton_per_fused_mean_native_layer_norm_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_native_layer_norm_13(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (256, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg2_1, (256, ), (1, ))
    assert_size_stride(arg3_1, (256, ), (1, ))
    assert_size_stride(arg4_1, (256, ), (1, ))
    assert_size_stride(arg5_1, (1536, 256), (256, 1))
    assert_size_stride(arg6_1, (1536, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (196, 196), (196, 1))
    assert_size_stride(arg10_1, (196, ), (1, ))
    assert_size_stride(arg11_1, (256, 768), (768, 1))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (1536, 256), (256, 1))
    assert_size_stride(arg16_1, (1536, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (196, 196), (196, 1))
    assert_size_stride(arg20_1, (196, ), (1, ))
    assert_size_stride(arg21_1, (256, 768), (768, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (1536, 256), (256, 1))
    assert_size_stride(arg26_1, (1536, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (196, 196), (196, 1))
    assert_size_stride(arg30_1, (196, ), (1, ))
    assert_size_stride(arg31_1, (256, 768), (768, 1))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (1536, 256), (256, 1))
    assert_size_stride(arg36_1, (1536, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (196, 196), (196, 1))
    assert_size_stride(arg40_1, (196, ), (1, ))
    assert_size_stride(arg41_1, (256, 768), (768, 1))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (1536, 256), (256, 1))
    assert_size_stride(arg46_1, (1536, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (196, 196), (196, 1))
    assert_size_stride(arg50_1, (196, ), (1, ))
    assert_size_stride(arg51_1, (256, 768), (768, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (256, ), (1, ))
    assert_size_stride(arg55_1, (1536, 256), (256, 1))
    assert_size_stride(arg56_1, (1536, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (196, 196), (196, 1))
    assert_size_stride(arg60_1, (196, ), (1, ))
    assert_size_stride(arg61_1, (256, 768), (768, 1))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (256, ), (1, ))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (1536, 256), (256, 1))
    assert_size_stride(arg66_1, (1536, ), (1, ))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (196, 196), (196, 1))
    assert_size_stride(arg70_1, (196, ), (1, ))
    assert_size_stride(arg71_1, (256, 768), (768, 1))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (1536, 256), (256, 1))
    assert_size_stride(arg76_1, (1536, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (196, 196), (196, 1))
    assert_size_stride(arg80_1, (196, ), (1, ))
    assert_size_stride(arg81_1, (256, 768), (768, 1))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (256, ), (1, ))
    assert_size_stride(arg84_1, (256, ), (1, ))
    assert_size_stride(arg85_1, (1536, 256), (256, 1))
    assert_size_stride(arg86_1, (1536, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (196, 196), (196, 1))
    assert_size_stride(arg90_1, (196, ), (1, ))
    assert_size_stride(arg91_1, (256, 768), (768, 1))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (1536, 256), (256, 1))
    assert_size_stride(arg96_1, (1536, ), (1, ))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (196, 196), (196, 1))
    assert_size_stride(arg100_1, (196, ), (1, ))
    assert_size_stride(arg101_1, (256, 768), (768, 1))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (256, ), (1, ))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (1536, 256), (256, 1))
    assert_size_stride(arg106_1, (1536, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (196, 196), (196, 1))
    assert_size_stride(arg110_1, (196, ), (1, ))
    assert_size_stride(arg111_1, (256, 768), (768, 1))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (256, ), (1, ))
    assert_size_stride(arg114_1, (256, ), (1, ))
    assert_size_stride(arg115_1, (1536, 256), (256, 1))
    assert_size_stride(arg116_1, (1536, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (196, 196), (196, 1))
    assert_size_stride(arg120_1, (196, ), (1, ))
    assert_size_stride(arg121_1, (256, 768), (768, 1))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (1536, 256), (256, 1))
    assert_size_stride(arg126_1, (1536, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (196, 196), (196, 1))
    assert_size_stride(arg130_1, (196, ), (1, ))
    assert_size_stride(arg131_1, (256, 768), (768, 1))
    assert_size_stride(arg132_1, (256, ), (1, ))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (256, ), (1, ))
    assert_size_stride(arg135_1, (1536, 256), (256, 1))
    assert_size_stride(arg136_1, (1536, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (196, 196), (196, 1))
    assert_size_stride(arg140_1, (196, ), (1, ))
    assert_size_stride(arg141_1, (256, 768), (768, 1))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (1536, 256), (256, 1))
    assert_size_stride(arg146_1, (1536, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (196, 196), (196, 1))
    assert_size_stride(arg150_1, (196, ), (1, ))
    assert_size_stride(arg151_1, (256, 768), (768, 1))
    assert_size_stride(arg152_1, (256, ), (1, ))
    assert_size_stride(arg153_1, (256, ), (1, ))
    assert_size_stride(arg154_1, (256, ), (1, ))
    assert_size_stride(arg155_1, (1536, 256), (256, 1))
    assert_size_stride(arg156_1, (1536, ), (1, ))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (196, 196), (196, 1))
    assert_size_stride(arg160_1, (196, ), (1, ))
    assert_size_stride(arg161_1, (256, 768), (768, 1))
    assert_size_stride(arg162_1, (256, ), (1, ))
    assert_size_stride(arg163_1, (256, ), (1, ))
    assert_size_stride(arg164_1, (256, ), (1, ))
    assert_size_stride(arg165_1, (1536, 256), (256, 1))
    assert_size_stride(arg166_1, (1536, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (196, 196), (196, 1))
    assert_size_stride(arg170_1, (196, ), (1, ))
    assert_size_stride(arg171_1, (256, 768), (768, 1))
    assert_size_stride(arg172_1, (256, ), (1, ))
    assert_size_stride(arg173_1, (256, ), (1, ))
    assert_size_stride(arg174_1, (256, ), (1, ))
    assert_size_stride(arg175_1, (1536, 256), (256, 1))
    assert_size_stride(arg176_1, (1536, ), (1, ))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (196, 196), (196, 1))
    assert_size_stride(arg180_1, (196, ), (1, ))
    assert_size_stride(arg181_1, (256, 768), (768, 1))
    assert_size_stride(arg182_1, (256, ), (1, ))
    assert_size_stride(arg183_1, (256, ), (1, ))
    assert_size_stride(arg184_1, (256, ), (1, ))
    assert_size_stride(arg185_1, (1536, 256), (256, 1))
    assert_size_stride(arg186_1, (1536, ), (1, ))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (196, 196), (196, 1))
    assert_size_stride(arg190_1, (196, ), (1, ))
    assert_size_stride(arg191_1, (256, 768), (768, 1))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (256, ), (1, ))
    assert_size_stride(arg194_1, (256, ), (1, ))
    assert_size_stride(arg195_1, (1536, 256), (256, 1))
    assert_size_stride(arg196_1, (1536, ), (1, ))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (768, ), (1, ))
    assert_size_stride(arg199_1, (196, 196), (196, 1))
    assert_size_stride(arg200_1, (196, ), (1, ))
    assert_size_stride(arg201_1, (256, 768), (768, 1))
    assert_size_stride(arg202_1, (256, ), (1, ))
    assert_size_stride(arg203_1, (256, ), (1, ))
    assert_size_stride(arg204_1, (256, ), (1, ))
    assert_size_stride(arg205_1, (1536, 256), (256, 1))
    assert_size_stride(arg206_1, (1536, ), (1, ))
    assert_size_stride(arg207_1, (768, ), (1, ))
    assert_size_stride(arg208_1, (768, ), (1, ))
    assert_size_stride(arg209_1, (196, 196), (196, 1))
    assert_size_stride(arg210_1, (196, ), (1, ))
    assert_size_stride(arg211_1, (256, 768), (768, 1))
    assert_size_stride(arg212_1, (256, ), (1, ))
    assert_size_stride(arg213_1, (256, ), (1, ))
    assert_size_stride(arg214_1, (256, ), (1, ))
    assert_size_stride(arg215_1, (1536, 256), (256, 1))
    assert_size_stride(arg216_1, (1536, ), (1, ))
    assert_size_stride(arg217_1, (768, ), (1, ))
    assert_size_stride(arg218_1, (768, ), (1, ))
    assert_size_stride(arg219_1, (196, 196), (196, 1))
    assert_size_stride(arg220_1, (196, ), (1, ))
    assert_size_stride(arg221_1, (256, 768), (768, 1))
    assert_size_stride(arg222_1, (256, ), (1, ))
    assert_size_stride(arg223_1, (256, ), (1, ))
    assert_size_stride(arg224_1, (256, ), (1, ))
    assert_size_stride(arg225_1, (1536, 256), (256, 1))
    assert_size_stride(arg226_1, (1536, ), (1, ))
    assert_size_stride(arg227_1, (768, ), (1, ))
    assert_size_stride(arg228_1, (768, ), (1, ))
    assert_size_stride(arg229_1, (196, 196), (196, 1))
    assert_size_stride(arg230_1, (196, ), (1, ))
    assert_size_stride(arg231_1, (256, 768), (768, 1))
    assert_size_stride(arg232_1, (256, ), (1, ))
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (256, ), (1, ))
    assert_size_stride(arg235_1, (1536, 256), (256, 1))
    assert_size_stride(arg236_1, (1536, ), (1, ))
    assert_size_stride(arg237_1, (768, ), (1, ))
    assert_size_stride(arg238_1, (768, ), (1, ))
    assert_size_stride(arg239_1, (196, 196), (196, 1))
    assert_size_stride(arg240_1, (196, ), (1, ))
    assert_size_stride(arg241_1, (256, 768), (768, 1))
    assert_size_stride(arg242_1, (256, ), (1, ))
    assert_size_stride(arg243_1, (256, ), (1, ))
    assert_size_stride(arg244_1, (256, ), (1, ))
    assert_size_stride(arg245_1, (1536, 256), (256, 1))
    assert_size_stride(arg246_1, (1536, ), (1, ))
    assert_size_stride(arg247_1, (768, ), (1, ))
    assert_size_stride(arg248_1, (768, ), (1, ))
    assert_size_stride(arg249_1, (196, 196), (196, 1))
    assert_size_stride(arg250_1, (196, ), (1, ))
    assert_size_stride(arg251_1, (256, 768), (768, 1))
    assert_size_stride(arg252_1, (256, ), (1, ))
    assert_size_stride(arg253_1, (256, ), (1, ))
    assert_size_stride(arg254_1, (256, ), (1, ))
    assert_size_stride(arg255_1, (1536, 256), (256, 1))
    assert_size_stride(arg256_1, (1536, ), (1, ))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (768, ), (1, ))
    assert_size_stride(arg259_1, (196, 196), (196, 1))
    assert_size_stride(arg260_1, (196, ), (1, ))
    assert_size_stride(arg261_1, (256, 768), (768, 1))
    assert_size_stride(arg262_1, (256, ), (1, ))
    assert_size_stride(arg263_1, (256, ), (1, ))
    assert_size_stride(arg264_1, (256, ), (1, ))
    assert_size_stride(arg265_1, (1536, 256), (256, 1))
    assert_size_stride(arg266_1, (1536, ), (1, ))
    assert_size_stride(arg267_1, (768, ), (1, ))
    assert_size_stride(arg268_1, (768, ), (1, ))
    assert_size_stride(arg269_1, (196, 196), (196, 1))
    assert_size_stride(arg270_1, (196, ), (1, ))
    assert_size_stride(arg271_1, (256, 768), (768, 1))
    assert_size_stride(arg272_1, (256, ), (1, ))
    assert_size_stride(arg273_1, (256, ), (1, ))
    assert_size_stride(arg274_1, (256, ), (1, ))
    assert_size_stride(arg275_1, (1536, 256), (256, 1))
    assert_size_stride(arg276_1, (1536, ), (1, ))
    assert_size_stride(arg277_1, (768, ), (1, ))
    assert_size_stride(arg278_1, (768, ), (1, ))
    assert_size_stride(arg279_1, (196, 196), (196, 1))
    assert_size_stride(arg280_1, (196, ), (1, ))
    assert_size_stride(arg281_1, (256, 768), (768, 1))
    assert_size_stride(arg282_1, (256, ), (1, ))
    assert_size_stride(arg283_1, (256, ), (1, ))
    assert_size_stride(arg284_1, (256, ), (1, ))
    assert_size_stride(arg285_1, (1536, 256), (256, 1))
    assert_size_stride(arg286_1, (1536, ), (1, ))
    assert_size_stride(arg287_1, (768, ), (1, ))
    assert_size_stride(arg288_1, (768, ), (1, ))
    assert_size_stride(arg289_1, (196, 196), (196, 1))
    assert_size_stride(arg290_1, (196, ), (1, ))
    assert_size_stride(arg291_1, (256, 768), (768, 1))
    assert_size_stride(arg292_1, (256, ), (1, ))
    assert_size_stride(arg293_1, (256, ), (1, ))
    assert_size_stride(arg294_1, (256, ), (1, ))
    assert_size_stride(arg295_1, (1536, 256), (256, 1))
    assert_size_stride(arg296_1, (1536, ), (1, ))
    assert_size_stride(arg297_1, (768, ), (1, ))
    assert_size_stride(arg298_1, (768, ), (1, ))
    assert_size_stride(arg299_1, (196, 196), (196, 1))
    assert_size_stride(arg300_1, (196, ), (1, ))
    assert_size_stride(arg301_1, (256, 768), (768, 1))
    assert_size_stride(arg302_1, (256, ), (1, ))
    assert_size_stride(arg303_1, (256, ), (1, ))
    assert_size_stride(arg304_1, (256, ), (1, ))
    assert_size_stride(arg305_1, (1000, 256), (256, 1))
    assert_size_stride(arg306_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_216], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((8, 196, 1, 2), (392, 1, 3136, 196), torch.float32)
        buf2 = empty_strided_cuda((8, 196, 1, 2), (392, 1, 3136, 196), torch.float32)
        buf3 = empty_strided_cuda((8, 196, 1, 2), (392, 1, 3136, 196), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_61], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf0, arg2_1, buf1, buf2, buf3, 3136, 128, grid=grid(3136), stream=stream0)
        buf4 = empty_strided_cuda((8, 196, 1), (196, 1, 1568), torch.float32)
        buf5 = empty_strided_cuda((8, 196, 1), (196, 1, 1568), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_61], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 1568, 2, grid=grid(1568), stream=stream0)
        buf7 = empty_strided_cuda((8, 196, 256), (50176, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_61], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_2.run(buf0, arg2_1, buf4, buf5, arg3_1, arg4_1, buf7, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del arg3_1
        del arg4_1
        buf8 = empty_strided_cuda((1568, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (1568, 256), (256, 1), 0), reinterpret_tensor(arg5_1, (256, 1536), (1, 256), 0), out=buf8)
        del arg5_1
        buf12 = empty_strided_cuda((8, 768, 196), (150528, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_91, v_92], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf8, arg6_1, arg7_1, arg8_1, buf12, 1568, 768, grid=grid(1568), stream=stream0)
        del arg7_1
        del arg8_1
        buf13 = empty_strided_cuda((6144, 196), (196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_92], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (6144, 196), (196, 1), 0), reinterpret_tensor(arg9_1, (196, 196), (1, 196), 0), out=buf13)
        del arg9_1
        buf14 = reinterpret_tensor(buf12, (8, 196, 768), (150528, 768, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf8, arg6_1, buf13, arg10_1, buf14, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg10_1
        del arg6_1
        buf15 = reinterpret_tensor(buf7, (1568, 256), (256, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf14, (1568, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 256), (1, 768), 0), out=buf15)
        del arg11_1
        buf16 = reinterpret_tensor(buf3, (8, 196, 1, 2), (392, 2, 3136, 1), 0); del buf3  # reuse
        buf17 = reinterpret_tensor(buf2, (8, 196, 1, 2), (392, 2, 3136, 1), 0); del buf2  # reuse
        buf18 = reinterpret_tensor(buf1, (8, 196, 1, 2), (392, 2, 3136, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_224, layer_norm_63], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf0, arg2_1, buf15, arg12_1, buf16, buf17, buf18, 3136, 128, grid=grid(3136), stream=stream0)
        buf19 = buf5; del buf5  # reuse
        buf20 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_224, layer_norm_63], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf16, buf17, buf18, buf19, buf20, 1568, 2, grid=grid(1568), stream=stream0)
        del buf16
        del buf17
        del buf18
        buf22 = empty_strided_cuda((8, 196, 256), (50176, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_224, layer_norm_63], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_7.run(buf0, arg2_1, buf15, arg12_1, buf19, buf20, arg13_1, arg14_1, buf22, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del arg13_1
        del arg14_1
        buf23 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (1568, 256), (256, 1), 0), reinterpret_tensor(arg15_1, (256, 1536), (1, 256), 0), out=buf23)
        del arg15_1
        buf27 = reinterpret_tensor(buf14, (8, 768, 196), (150528, 196, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [v_94, v_95], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf23, arg16_1, arg17_1, arg18_1, buf27, 1568, 768, grid=grid(1568), stream=stream0)
        del arg17_1
        del arg18_1
        buf28 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [v_95], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (6144, 196), (196, 1), 0), reinterpret_tensor(arg19_1, (196, 196), (1, 196), 0), out=buf28)
        del arg19_1
        buf29 = reinterpret_tensor(buf27, (8, 196, 768), (150528, 768, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [x_228], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf23, arg16_1, buf28, arg20_1, buf29, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg16_1
        del arg20_1
        buf30 = reinterpret_tensor(buf22, (1568, 256), (256, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (1568, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 256), (1, 768), 0), out=buf30)
        del arg21_1
        buf31 = reinterpret_tensor(buf15, (8, 196, 256), (50176, 256, 1), 0); del buf15  # reuse
        buf35 = empty_strided_cuda((8, 196, 256), (50176, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_224, x_231, layer_norm_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf31, buf0, arg2_1, arg12_1, buf30, arg22_1, arg23_1, arg24_1, buf35, 1568, 256, grid=grid(1568), stream=stream0)
        del arg12_1
        del arg22_1
        del arg23_1
        del arg24_1
        del arg2_1
        buf36 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf35, (1568, 256), (256, 1), 0), reinterpret_tensor(arg25_1, (256, 1536), (1, 256), 0), out=buf36)
        del arg25_1
        buf40 = reinterpret_tensor(buf29, (8, 768, 196), (150528, 196, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [v_97, v_98], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf36, arg26_1, arg27_1, arg28_1, buf40, 1568, 768, grid=grid(1568), stream=stream0)
        del arg27_1
        del arg28_1
        buf41 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [v_98], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (6144, 196), (196, 1), 0), reinterpret_tensor(arg29_1, (196, 196), (1, 196), 0), out=buf41)
        del arg29_1
        buf42 = reinterpret_tensor(buf40, (8, 196, 768), (150528, 768, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_235], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf36, arg26_1, buf41, arg30_1, buf42, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg26_1
        del arg30_1
        buf43 = reinterpret_tensor(buf35, (1568, 256), (256, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (1568, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 256), (1, 768), 0), out=buf43)
        del arg31_1
        buf47 = reinterpret_tensor(buf30, (8, 196, 256), (50176, 256, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_238, layer_norm_67], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf31, buf43, arg32_1, arg33_1, arg34_1, buf47, 1568, 256, grid=grid(1568), stream=stream0)
        del arg33_1
        del arg34_1
        buf48 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (1568, 256), (256, 1), 0), reinterpret_tensor(arg35_1, (256, 1536), (1, 256), 0), out=buf48)
        del arg35_1
        buf52 = reinterpret_tensor(buf42, (8, 768, 196), (150528, 196, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [v_100, v_101], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf48, arg36_1, arg37_1, arg38_1, buf52, 1568, 768, grid=grid(1568), stream=stream0)
        del arg37_1
        del arg38_1
        buf53 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [v_101], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (6144, 196), (196, 1), 0), reinterpret_tensor(arg39_1, (196, 196), (1, 196), 0), out=buf53)
        del arg39_1
        buf54 = reinterpret_tensor(buf52, (8, 196, 768), (150528, 768, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_242], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf48, arg36_1, buf53, arg40_1, buf54, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg36_1
        del arg40_1
        buf55 = reinterpret_tensor(buf47, (1568, 256), (256, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf54, (1568, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 256), (1, 768), 0), out=buf55)
        del arg41_1
        buf56 = reinterpret_tensor(buf55, (8, 196, 256), (50176, 256, 1), 0); del buf55  # reuse
        buf60 = reinterpret_tensor(buf0, (8, 196, 256), (50176, 256, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_238, x_245, layer_norm_69], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf56, buf31, buf43, arg32_1, arg42_1, arg43_1, arg44_1, buf60, 1568, 256, grid=grid(1568), stream=stream0)
        del arg32_1
        del arg42_1
        del arg43_1
        del arg44_1
        buf61 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (1568, 256), (256, 1), 0), reinterpret_tensor(arg45_1, (256, 1536), (1, 256), 0), out=buf61)
        del arg45_1
        buf65 = reinterpret_tensor(buf54, (8, 768, 196), (150528, 196, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [v_103, v_104], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf61, arg46_1, arg47_1, arg48_1, buf65, 1568, 768, grid=grid(1568), stream=stream0)
        del arg47_1
        del arg48_1
        buf66 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [v_104], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (6144, 196), (196, 1), 0), reinterpret_tensor(arg49_1, (196, 196), (1, 196), 0), out=buf66)
        del arg49_1
        buf67 = reinterpret_tensor(buf65, (8, 196, 768), (150528, 768, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_249], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf61, arg46_1, buf66, arg50_1, buf67, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg46_1
        del arg50_1
        buf68 = reinterpret_tensor(buf60, (1568, 256), (256, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (1568, 768), (768, 1), 0), reinterpret_tensor(arg51_1, (768, 256), (1, 768), 0), out=buf68)
        del arg51_1
        buf72 = reinterpret_tensor(buf43, (8, 196, 256), (50176, 256, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_252, layer_norm_71], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf56, buf68, arg52_1, arg53_1, arg54_1, buf72, 1568, 256, grid=grid(1568), stream=stream0)
        del arg53_1
        del arg54_1
        buf73 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (1568, 256), (256, 1), 0), reinterpret_tensor(arg55_1, (256, 1536), (1, 256), 0), out=buf73)
        del arg55_1
        buf77 = reinterpret_tensor(buf67, (8, 768, 196), (150528, 196, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [v_106, v_107], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf73, arg56_1, arg57_1, arg58_1, buf77, 1568, 768, grid=grid(1568), stream=stream0)
        del arg57_1
        del arg58_1
        buf78 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [v_107], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (6144, 196), (196, 1), 0), reinterpret_tensor(arg59_1, (196, 196), (1, 196), 0), out=buf78)
        del arg59_1
        buf79 = reinterpret_tensor(buf77, (8, 196, 768), (150528, 768, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [x_256], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf73, arg56_1, buf78, arg60_1, buf79, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg56_1
        del arg60_1
        buf80 = reinterpret_tensor(buf72, (1568, 256), (256, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (1568, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 256), (1, 768), 0), out=buf80)
        del arg61_1
        buf81 = reinterpret_tensor(buf80, (8, 196, 256), (50176, 256, 1), 0); del buf80  # reuse
        buf85 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_252, x_259, layer_norm_73], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf81, buf56, buf68, arg52_1, arg62_1, arg63_1, arg64_1, buf85, 1568, 256, grid=grid(1568), stream=stream0)
        del arg52_1
        del arg62_1
        del arg63_1
        del arg64_1
        buf86 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf85, (1568, 256), (256, 1), 0), reinterpret_tensor(arg65_1, (256, 1536), (1, 256), 0), out=buf86)
        del arg65_1
        buf90 = reinterpret_tensor(buf79, (8, 768, 196), (150528, 196, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [v_109, v_110], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf86, arg66_1, arg67_1, arg68_1, buf90, 1568, 768, grid=grid(1568), stream=stream0)
        del arg67_1
        del arg68_1
        buf91 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [v_110], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (6144, 196), (196, 1), 0), reinterpret_tensor(arg69_1, (196, 196), (1, 196), 0), out=buf91)
        del arg69_1
        buf92 = reinterpret_tensor(buf90, (8, 196, 768), (150528, 768, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_263], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf86, arg66_1, buf91, arg70_1, buf92, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg66_1
        del arg70_1
        buf93 = reinterpret_tensor(buf85, (1568, 256), (256, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (1568, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 256), (1, 768), 0), out=buf93)
        del arg71_1
        buf97 = reinterpret_tensor(buf68, (8, 196, 256), (50176, 256, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_266, layer_norm_75], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf81, buf93, arg72_1, arg73_1, arg74_1, buf97, 1568, 256, grid=grid(1568), stream=stream0)
        del arg73_1
        del arg74_1
        buf98 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (1568, 256), (256, 1), 0), reinterpret_tensor(arg75_1, (256, 1536), (1, 256), 0), out=buf98)
        del arg75_1
        buf102 = reinterpret_tensor(buf92, (8, 768, 196), (150528, 196, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [v_112, v_113], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf98, arg76_1, arg77_1, arg78_1, buf102, 1568, 768, grid=grid(1568), stream=stream0)
        del arg77_1
        del arg78_1
        buf103 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [v_113], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (6144, 196), (196, 1), 0), reinterpret_tensor(arg79_1, (196, 196), (1, 196), 0), out=buf103)
        del arg79_1
        buf104 = reinterpret_tensor(buf102, (8, 196, 768), (150528, 768, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_270], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf98, arg76_1, buf103, arg80_1, buf104, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg76_1
        del arg80_1
        buf105 = reinterpret_tensor(buf97, (1568, 256), (256, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf104, (1568, 768), (768, 1), 0), reinterpret_tensor(arg81_1, (768, 256), (1, 768), 0), out=buf105)
        del arg81_1
        buf106 = reinterpret_tensor(buf105, (8, 196, 256), (50176, 256, 1), 0); del buf105  # reuse
        buf110 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_266, x_273, layer_norm_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf106, buf81, buf93, arg72_1, arg82_1, arg83_1, arg84_1, buf110, 1568, 256, grid=grid(1568), stream=stream0)
        del arg72_1
        del arg82_1
        del arg83_1
        del arg84_1
        buf111 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (1568, 256), (256, 1), 0), reinterpret_tensor(arg85_1, (256, 1536), (1, 256), 0), out=buf111)
        del arg85_1
        buf115 = reinterpret_tensor(buf104, (8, 768, 196), (150528, 196, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [v_115, v_116], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf111, arg86_1, arg87_1, arg88_1, buf115, 1568, 768, grid=grid(1568), stream=stream0)
        del arg87_1
        del arg88_1
        buf116 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [v_116], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (6144, 196), (196, 1), 0), reinterpret_tensor(arg89_1, (196, 196), (1, 196), 0), out=buf116)
        del arg89_1
        buf117 = reinterpret_tensor(buf115, (8, 196, 768), (150528, 768, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_277], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf111, arg86_1, buf116, arg90_1, buf117, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg86_1
        del arg90_1
        buf118 = reinterpret_tensor(buf110, (1568, 256), (256, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf117, (1568, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 256), (1, 768), 0), out=buf118)
        del arg91_1
        buf122 = reinterpret_tensor(buf93, (8, 196, 256), (50176, 256, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_280, layer_norm_79], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf106, buf118, arg92_1, arg93_1, arg94_1, buf122, 1568, 256, grid=grid(1568), stream=stream0)
        del arg93_1
        del arg94_1
        buf123 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (1568, 256), (256, 1), 0), reinterpret_tensor(arg95_1, (256, 1536), (1, 256), 0), out=buf123)
        del arg95_1
        buf127 = reinterpret_tensor(buf117, (8, 768, 196), (150528, 196, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [v_118, v_119], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf123, arg96_1, arg97_1, arg98_1, buf127, 1568, 768, grid=grid(1568), stream=stream0)
        del arg97_1
        del arg98_1
        buf128 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [v_119], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (6144, 196), (196, 1), 0), reinterpret_tensor(arg99_1, (196, 196), (1, 196), 0), out=buf128)
        del arg99_1
        buf129 = reinterpret_tensor(buf127, (8, 196, 768), (150528, 768, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_284], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf123, arg96_1, buf128, arg100_1, buf129, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg100_1
        del arg96_1
        buf130 = reinterpret_tensor(buf122, (1568, 256), (256, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (1568, 768), (768, 1), 0), reinterpret_tensor(arg101_1, (768, 256), (1, 768), 0), out=buf130)
        del arg101_1
        buf131 = reinterpret_tensor(buf130, (8, 196, 256), (50176, 256, 1), 0); del buf130  # reuse
        buf135 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_280, x_287, layer_norm_81], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf131, buf106, buf118, arg92_1, arg102_1, arg103_1, arg104_1, buf135, 1568, 256, grid=grid(1568), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg92_1
        buf136 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (1568, 256), (256, 1), 0), reinterpret_tensor(arg105_1, (256, 1536), (1, 256), 0), out=buf136)
        del arg105_1
        buf140 = reinterpret_tensor(buf129, (8, 768, 196), (150528, 196, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [v_121, v_122], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf136, arg106_1, arg107_1, arg108_1, buf140, 1568, 768, grid=grid(1568), stream=stream0)
        del arg107_1
        del arg108_1
        buf141 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [v_122], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (6144, 196), (196, 1), 0), reinterpret_tensor(arg109_1, (196, 196), (1, 196), 0), out=buf141)
        del arg109_1
        buf142 = reinterpret_tensor(buf140, (8, 196, 768), (150528, 768, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf136, arg106_1, buf141, arg110_1, buf142, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg106_1
        del arg110_1
        buf143 = reinterpret_tensor(buf135, (1568, 256), (256, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (1568, 768), (768, 1), 0), reinterpret_tensor(arg111_1, (768, 256), (1, 768), 0), out=buf143)
        del arg111_1
        buf147 = reinterpret_tensor(buf118, (8, 196, 256), (50176, 256, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_294, layer_norm_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf131, buf143, arg112_1, arg113_1, arg114_1, buf147, 1568, 256, grid=grid(1568), stream=stream0)
        del arg113_1
        del arg114_1
        buf148 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (1568, 256), (256, 1), 0), reinterpret_tensor(arg115_1, (256, 1536), (1, 256), 0), out=buf148)
        del arg115_1
        buf152 = reinterpret_tensor(buf142, (8, 768, 196), (150528, 196, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [v_124, v_125], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf148, arg116_1, arg117_1, arg118_1, buf152, 1568, 768, grid=grid(1568), stream=stream0)
        del arg117_1
        del arg118_1
        buf153 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [v_125], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (6144, 196), (196, 1), 0), reinterpret_tensor(arg119_1, (196, 196), (1, 196), 0), out=buf153)
        del arg119_1
        buf154 = reinterpret_tensor(buf152, (8, 196, 768), (150528, 768, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [x_298], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf148, arg116_1, buf153, arg120_1, buf154, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg116_1
        del arg120_1
        buf155 = reinterpret_tensor(buf147, (1568, 256), (256, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf154, (1568, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 256), (1, 768), 0), out=buf155)
        del arg121_1
        buf156 = reinterpret_tensor(buf155, (8, 196, 256), (50176, 256, 1), 0); del buf155  # reuse
        buf160 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_294, x_301, layer_norm_85], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf156, buf131, buf143, arg112_1, arg122_1, arg123_1, arg124_1, buf160, 1568, 256, grid=grid(1568), stream=stream0)
        del arg112_1
        del arg122_1
        del arg123_1
        del arg124_1
        buf161 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (1568, 256), (256, 1), 0), reinterpret_tensor(arg125_1, (256, 1536), (1, 256), 0), out=buf161)
        del arg125_1
        buf165 = reinterpret_tensor(buf154, (8, 768, 196), (150528, 196, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [v_127, v_128], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf161, arg126_1, arg127_1, arg128_1, buf165, 1568, 768, grid=grid(1568), stream=stream0)
        del arg127_1
        del arg128_1
        buf166 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [v_128], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (6144, 196), (196, 1), 0), reinterpret_tensor(arg129_1, (196, 196), (1, 196), 0), out=buf166)
        del arg129_1
        buf167 = reinterpret_tensor(buf165, (8, 196, 768), (150528, 768, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_305], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf161, arg126_1, buf166, arg130_1, buf167, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg126_1
        del arg130_1
        buf168 = reinterpret_tensor(buf160, (1568, 256), (256, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (1568, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 256), (1, 768), 0), out=buf168)
        del arg131_1
        buf172 = reinterpret_tensor(buf143, (8, 196, 256), (50176, 256, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_308, layer_norm_87], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf156, buf168, arg132_1, arg133_1, arg134_1, buf172, 1568, 256, grid=grid(1568), stream=stream0)
        del arg133_1
        del arg134_1
        buf173 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (1568, 256), (256, 1), 0), reinterpret_tensor(arg135_1, (256, 1536), (1, 256), 0), out=buf173)
        del arg135_1
        buf177 = reinterpret_tensor(buf167, (8, 768, 196), (150528, 196, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [v_130, v_131], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf173, arg136_1, arg137_1, arg138_1, buf177, 1568, 768, grid=grid(1568), stream=stream0)
        del arg137_1
        del arg138_1
        buf178 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [v_131], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (6144, 196), (196, 1), 0), reinterpret_tensor(arg139_1, (196, 196), (1, 196), 0), out=buf178)
        del arg139_1
        buf179 = reinterpret_tensor(buf177, (8, 196, 768), (150528, 768, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_312], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf173, arg136_1, buf178, arg140_1, buf179, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg136_1
        del arg140_1
        buf180 = reinterpret_tensor(buf172, (1568, 256), (256, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (1568, 768), (768, 1), 0), reinterpret_tensor(arg141_1, (768, 256), (1, 768), 0), out=buf180)
        del arg141_1
        buf181 = reinterpret_tensor(buf180, (8, 196, 256), (50176, 256, 1), 0); del buf180  # reuse
        buf185 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_308, x_315, layer_norm_89], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf181, buf156, buf168, arg132_1, arg142_1, arg143_1, arg144_1, buf185, 1568, 256, grid=grid(1568), stream=stream0)
        del arg132_1
        del arg142_1
        del arg143_1
        del arg144_1
        buf186 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf185, (1568, 256), (256, 1), 0), reinterpret_tensor(arg145_1, (256, 1536), (1, 256), 0), out=buf186)
        del arg145_1
        buf190 = reinterpret_tensor(buf179, (8, 768, 196), (150528, 196, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [v_133, v_134], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf186, arg146_1, arg147_1, arg148_1, buf190, 1568, 768, grid=grid(1568), stream=stream0)
        del arg147_1
        del arg148_1
        buf191 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [v_134], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (6144, 196), (196, 1), 0), reinterpret_tensor(arg149_1, (196, 196), (1, 196), 0), out=buf191)
        del arg149_1
        buf192 = reinterpret_tensor(buf190, (8, 196, 768), (150528, 768, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_319], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf186, arg146_1, buf191, arg150_1, buf192, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg146_1
        del arg150_1
        buf193 = reinterpret_tensor(buf185, (1568, 256), (256, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf192, (1568, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 256), (1, 768), 0), out=buf193)
        del arg151_1
        buf197 = reinterpret_tensor(buf168, (8, 196, 256), (50176, 256, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_322, layer_norm_91], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf181, buf193, arg152_1, arg153_1, arg154_1, buf197, 1568, 256, grid=grid(1568), stream=stream0)
        del arg153_1
        del arg154_1
        buf198 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (1568, 256), (256, 1), 0), reinterpret_tensor(arg155_1, (256, 1536), (1, 256), 0), out=buf198)
        del arg155_1
        buf202 = reinterpret_tensor(buf192, (8, 768, 196), (150528, 196, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [v_136, v_137], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf198, arg156_1, arg157_1, arg158_1, buf202, 1568, 768, grid=grid(1568), stream=stream0)
        del arg157_1
        del arg158_1
        buf203 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [v_137], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (6144, 196), (196, 1), 0), reinterpret_tensor(arg159_1, (196, 196), (1, 196), 0), out=buf203)
        del arg159_1
        buf204 = reinterpret_tensor(buf202, (8, 196, 768), (150528, 768, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [x_326], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf198, arg156_1, buf203, arg160_1, buf204, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg156_1
        del arg160_1
        buf205 = reinterpret_tensor(buf197, (1568, 256), (256, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf204, (1568, 768), (768, 1), 0), reinterpret_tensor(arg161_1, (768, 256), (1, 768), 0), out=buf205)
        del arg161_1
        buf206 = reinterpret_tensor(buf205, (8, 196, 256), (50176, 256, 1), 0); del buf205  # reuse
        buf210 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [x_322, x_329, layer_norm_93], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf206, buf181, buf193, arg152_1, arg162_1, arg163_1, arg164_1, buf210, 1568, 256, grid=grid(1568), stream=stream0)
        del arg152_1
        del arg162_1
        del arg163_1
        del arg164_1
        buf211 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (1568, 256), (256, 1), 0), reinterpret_tensor(arg165_1, (256, 1536), (1, 256), 0), out=buf211)
        del arg165_1
        buf215 = reinterpret_tensor(buf204, (8, 768, 196), (150528, 196, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [v_139, v_140], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf211, arg166_1, arg167_1, arg168_1, buf215, 1568, 768, grid=grid(1568), stream=stream0)
        del arg167_1
        del arg168_1
        buf216 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [v_140], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (6144, 196), (196, 1), 0), reinterpret_tensor(arg169_1, (196, 196), (1, 196), 0), out=buf216)
        del arg169_1
        buf217 = reinterpret_tensor(buf215, (8, 196, 768), (150528, 768, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [x_333], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf211, arg166_1, buf216, arg170_1, buf217, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg166_1
        del arg170_1
        buf218 = reinterpret_tensor(buf210, (1568, 256), (256, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (1568, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 256), (1, 768), 0), out=buf218)
        del arg171_1
        buf222 = reinterpret_tensor(buf193, (8, 196, 256), (50176, 256, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [x_336, layer_norm_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf206, buf218, arg172_1, arg173_1, arg174_1, buf222, 1568, 256, grid=grid(1568), stream=stream0)
        del arg173_1
        del arg174_1
        buf223 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf222, (1568, 256), (256, 1), 0), reinterpret_tensor(arg175_1, (256, 1536), (1, 256), 0), out=buf223)
        del arg175_1
        buf227 = reinterpret_tensor(buf217, (8, 768, 196), (150528, 196, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [v_142, v_143], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf223, arg176_1, arg177_1, arg178_1, buf227, 1568, 768, grid=grid(1568), stream=stream0)
        del arg177_1
        del arg178_1
        buf228 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [v_143], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (6144, 196), (196, 1), 0), reinterpret_tensor(arg179_1, (196, 196), (1, 196), 0), out=buf228)
        del arg179_1
        buf229 = reinterpret_tensor(buf227, (8, 196, 768), (150528, 768, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [x_340], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf223, arg176_1, buf228, arg180_1, buf229, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg176_1
        del arg180_1
        buf230 = reinterpret_tensor(buf222, (1568, 256), (256, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (1568, 768), (768, 1), 0), reinterpret_tensor(arg181_1, (768, 256), (1, 768), 0), out=buf230)
        del arg181_1
        buf231 = reinterpret_tensor(buf230, (8, 196, 256), (50176, 256, 1), 0); del buf230  # reuse
        buf235 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [x_336, x_343, layer_norm_97], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf231, buf206, buf218, arg172_1, arg182_1, arg183_1, arg184_1, buf235, 1568, 256, grid=grid(1568), stream=stream0)
        del arg172_1
        del arg182_1
        del arg183_1
        del arg184_1
        buf236 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (1568, 256), (256, 1), 0), reinterpret_tensor(arg185_1, (256, 1536), (1, 256), 0), out=buf236)
        del arg185_1
        buf240 = reinterpret_tensor(buf229, (8, 768, 196), (150528, 196, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [v_145, v_146], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf236, arg186_1, arg187_1, arg188_1, buf240, 1568, 768, grid=grid(1568), stream=stream0)
        del arg187_1
        del arg188_1
        buf241 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [v_146], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (6144, 196), (196, 1), 0), reinterpret_tensor(arg189_1, (196, 196), (1, 196), 0), out=buf241)
        del arg189_1
        buf242 = reinterpret_tensor(buf240, (8, 196, 768), (150528, 768, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [x_347], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf236, arg186_1, buf241, arg190_1, buf242, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg186_1
        del arg190_1
        buf243 = reinterpret_tensor(buf235, (1568, 256), (256, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (1568, 768), (768, 1), 0), reinterpret_tensor(arg191_1, (768, 256), (1, 768), 0), out=buf243)
        del arg191_1
        buf247 = reinterpret_tensor(buf218, (8, 196, 256), (50176, 256, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [x_350, layer_norm_99], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf231, buf243, arg192_1, arg193_1, arg194_1, buf247, 1568, 256, grid=grid(1568), stream=stream0)
        del arg193_1
        del arg194_1
        buf248 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (1568, 256), (256, 1), 0), reinterpret_tensor(arg195_1, (256, 1536), (1, 256), 0), out=buf248)
        del arg195_1
        buf252 = reinterpret_tensor(buf242, (8, 768, 196), (150528, 196, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [v_148, v_149], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf248, arg196_1, arg197_1, arg198_1, buf252, 1568, 768, grid=grid(1568), stream=stream0)
        del arg197_1
        del arg198_1
        buf253 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [v_149], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (6144, 196), (196, 1), 0), reinterpret_tensor(arg199_1, (196, 196), (1, 196), 0), out=buf253)
        del arg199_1
        buf254 = reinterpret_tensor(buf252, (8, 196, 768), (150528, 768, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [x_354], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf248, arg196_1, buf253, arg200_1, buf254, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg196_1
        del arg200_1
        buf255 = reinterpret_tensor(buf247, (1568, 256), (256, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf254, (1568, 768), (768, 1), 0), reinterpret_tensor(arg201_1, (768, 256), (1, 768), 0), out=buf255)
        del arg201_1
        buf256 = reinterpret_tensor(buf255, (8, 196, 256), (50176, 256, 1), 0); del buf255  # reuse
        buf260 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [x_350, x_357, layer_norm_101], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf256, buf231, buf243, arg192_1, arg202_1, arg203_1, arg204_1, buf260, 1568, 256, grid=grid(1568), stream=stream0)
        del arg192_1
        del arg202_1
        del arg203_1
        del arg204_1
        buf261 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (1568, 256), (256, 1), 0), reinterpret_tensor(arg205_1, (256, 1536), (1, 256), 0), out=buf261)
        del arg205_1
        buf265 = reinterpret_tensor(buf254, (8, 768, 196), (150528, 196, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [v_151, v_152], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf261, arg206_1, arg207_1, arg208_1, buf265, 1568, 768, grid=grid(1568), stream=stream0)
        del arg207_1
        del arg208_1
        buf266 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [v_152], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (6144, 196), (196, 1), 0), reinterpret_tensor(arg209_1, (196, 196), (1, 196), 0), out=buf266)
        del arg209_1
        buf267 = reinterpret_tensor(buf265, (8, 196, 768), (150528, 768, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [x_361], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf261, arg206_1, buf266, arg210_1, buf267, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg206_1
        del arg210_1
        buf268 = reinterpret_tensor(buf260, (1568, 256), (256, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (1568, 768), (768, 1), 0), reinterpret_tensor(arg211_1, (768, 256), (1, 768), 0), out=buf268)
        del arg211_1
        buf272 = reinterpret_tensor(buf243, (8, 196, 256), (50176, 256, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [x_364, layer_norm_103], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf256, buf268, arg212_1, arg213_1, arg214_1, buf272, 1568, 256, grid=grid(1568), stream=stream0)
        del arg213_1
        del arg214_1
        buf273 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (1568, 256), (256, 1), 0), reinterpret_tensor(arg215_1, (256, 1536), (1, 256), 0), out=buf273)
        del arg215_1
        buf277 = reinterpret_tensor(buf267, (8, 768, 196), (150528, 196, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [v_154, v_155], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf273, arg216_1, arg217_1, arg218_1, buf277, 1568, 768, grid=grid(1568), stream=stream0)
        del arg217_1
        del arg218_1
        buf278 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [v_155], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (6144, 196), (196, 1), 0), reinterpret_tensor(arg219_1, (196, 196), (1, 196), 0), out=buf278)
        del arg219_1
        buf279 = reinterpret_tensor(buf277, (8, 196, 768), (150528, 768, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [x_368], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf273, arg216_1, buf278, arg220_1, buf279, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg216_1
        del arg220_1
        buf280 = reinterpret_tensor(buf272, (1568, 256), (256, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf279, (1568, 768), (768, 1), 0), reinterpret_tensor(arg221_1, (768, 256), (1, 768), 0), out=buf280)
        del arg221_1
        buf281 = reinterpret_tensor(buf280, (8, 196, 256), (50176, 256, 1), 0); del buf280  # reuse
        buf285 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [x_364, x_371, layer_norm_105], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf281, buf256, buf268, arg212_1, arg222_1, arg223_1, arg224_1, buf285, 1568, 256, grid=grid(1568), stream=stream0)
        del arg212_1
        del arg222_1
        del arg223_1
        del arg224_1
        buf286 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf285, (1568, 256), (256, 1), 0), reinterpret_tensor(arg225_1, (256, 1536), (1, 256), 0), out=buf286)
        del arg225_1
        buf290 = reinterpret_tensor(buf279, (8, 768, 196), (150528, 196, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [v_157, v_158], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf286, arg226_1, arg227_1, arg228_1, buf290, 1568, 768, grid=grid(1568), stream=stream0)
        del arg227_1
        del arg228_1
        buf291 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [v_158], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (6144, 196), (196, 1), 0), reinterpret_tensor(arg229_1, (196, 196), (1, 196), 0), out=buf291)
        del arg229_1
        buf292 = reinterpret_tensor(buf290, (8, 196, 768), (150528, 768, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [x_375], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf286, arg226_1, buf291, arg230_1, buf292, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg226_1
        del arg230_1
        buf293 = reinterpret_tensor(buf285, (1568, 256), (256, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (1568, 768), (768, 1), 0), reinterpret_tensor(arg231_1, (768, 256), (1, 768), 0), out=buf293)
        del arg231_1
        buf297 = reinterpret_tensor(buf268, (8, 196, 256), (50176, 256, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [x_378, layer_norm_107], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf281, buf293, arg232_1, arg233_1, arg234_1, buf297, 1568, 256, grid=grid(1568), stream=stream0)
        del arg233_1
        del arg234_1
        buf298 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf297, (1568, 256), (256, 1), 0), reinterpret_tensor(arg235_1, (256, 1536), (1, 256), 0), out=buf298)
        del arg235_1
        buf302 = reinterpret_tensor(buf292, (8, 768, 196), (150528, 196, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [v_160, v_161], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf298, arg236_1, arg237_1, arg238_1, buf302, 1568, 768, grid=grid(1568), stream=stream0)
        del arg237_1
        del arg238_1
        buf303 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [v_161], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf302, (6144, 196), (196, 1), 0), reinterpret_tensor(arg239_1, (196, 196), (1, 196), 0), out=buf303)
        del arg239_1
        buf304 = reinterpret_tensor(buf302, (8, 196, 768), (150528, 768, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [x_382], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf298, arg236_1, buf303, arg240_1, buf304, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg236_1
        del arg240_1
        buf305 = reinterpret_tensor(buf297, (1568, 256), (256, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (1568, 768), (768, 1), 0), reinterpret_tensor(arg241_1, (768, 256), (1, 768), 0), out=buf305)
        del arg241_1
        buf306 = reinterpret_tensor(buf305, (8, 196, 256), (50176, 256, 1), 0); del buf305  # reuse
        buf310 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [x_378, x_385, layer_norm_109], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf306, buf281, buf293, arg232_1, arg242_1, arg243_1, arg244_1, buf310, 1568, 256, grid=grid(1568), stream=stream0)
        del arg232_1
        del arg242_1
        del arg243_1
        del arg244_1
        buf311 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (1568, 256), (256, 1), 0), reinterpret_tensor(arg245_1, (256, 1536), (1, 256), 0), out=buf311)
        del arg245_1
        buf315 = reinterpret_tensor(buf304, (8, 768, 196), (150528, 196, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [v_163, v_164], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf311, arg246_1, arg247_1, arg248_1, buf315, 1568, 768, grid=grid(1568), stream=stream0)
        del arg247_1
        del arg248_1
        buf316 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [v_164], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (6144, 196), (196, 1), 0), reinterpret_tensor(arg249_1, (196, 196), (1, 196), 0), out=buf316)
        del arg249_1
        buf317 = reinterpret_tensor(buf315, (8, 196, 768), (150528, 768, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [x_389], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf311, arg246_1, buf316, arg250_1, buf317, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg246_1
        del arg250_1
        buf318 = reinterpret_tensor(buf310, (1568, 256), (256, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (1568, 768), (768, 1), 0), reinterpret_tensor(arg251_1, (768, 256), (1, 768), 0), out=buf318)
        del arg251_1
        buf322 = reinterpret_tensor(buf293, (8, 196, 256), (50176, 256, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [x_392, layer_norm_111], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf306, buf318, arg252_1, arg253_1, arg254_1, buf322, 1568, 256, grid=grid(1568), stream=stream0)
        del arg253_1
        del arg254_1
        buf323 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf322, (1568, 256), (256, 1), 0), reinterpret_tensor(arg255_1, (256, 1536), (1, 256), 0), out=buf323)
        del arg255_1
        buf327 = reinterpret_tensor(buf317, (8, 768, 196), (150528, 196, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [v_166, v_167], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf323, arg256_1, arg257_1, arg258_1, buf327, 1568, 768, grid=grid(1568), stream=stream0)
        del arg257_1
        del arg258_1
        buf328 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [v_167], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (6144, 196), (196, 1), 0), reinterpret_tensor(arg259_1, (196, 196), (1, 196), 0), out=buf328)
        del arg259_1
        buf329 = reinterpret_tensor(buf327, (8, 196, 768), (150528, 768, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [x_396], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf323, arg256_1, buf328, arg260_1, buf329, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg256_1
        del arg260_1
        buf330 = reinterpret_tensor(buf322, (1568, 256), (256, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf329, (1568, 768), (768, 1), 0), reinterpret_tensor(arg261_1, (768, 256), (1, 768), 0), out=buf330)
        del arg261_1
        buf331 = reinterpret_tensor(buf330, (8, 196, 256), (50176, 256, 1), 0); del buf330  # reuse
        buf335 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [x_392, x_399, layer_norm_113], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf331, buf306, buf318, arg252_1, arg262_1, arg263_1, arg264_1, buf335, 1568, 256, grid=grid(1568), stream=stream0)
        del arg252_1
        del arg262_1
        del arg263_1
        del arg264_1
        buf336 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1568, 256), (256, 1), 0), reinterpret_tensor(arg265_1, (256, 1536), (1, 256), 0), out=buf336)
        del arg265_1
        buf340 = reinterpret_tensor(buf329, (8, 768, 196), (150528, 196, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [v_169, v_170], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf336, arg266_1, arg267_1, arg268_1, buf340, 1568, 768, grid=grid(1568), stream=stream0)
        del arg267_1
        del arg268_1
        buf341 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [v_170], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (6144, 196), (196, 1), 0), reinterpret_tensor(arg269_1, (196, 196), (1, 196), 0), out=buf341)
        del arg269_1
        buf342 = reinterpret_tensor(buf340, (8, 196, 768), (150528, 768, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [x_403], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf336, arg266_1, buf341, arg270_1, buf342, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg266_1
        del arg270_1
        buf343 = reinterpret_tensor(buf335, (1568, 256), (256, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf342, (1568, 768), (768, 1), 0), reinterpret_tensor(arg271_1, (768, 256), (1, 768), 0), out=buf343)
        del arg271_1
        buf347 = reinterpret_tensor(buf318, (8, 196, 256), (50176, 256, 1), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [x_406, layer_norm_115], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf331, buf343, arg272_1, arg273_1, arg274_1, buf347, 1568, 256, grid=grid(1568), stream=stream0)
        del arg273_1
        del arg274_1
        buf348 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf347, (1568, 256), (256, 1), 0), reinterpret_tensor(arg275_1, (256, 1536), (1, 256), 0), out=buf348)
        del arg275_1
        buf352 = reinterpret_tensor(buf342, (8, 768, 196), (150528, 196, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [v_172, v_173], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf348, arg276_1, arg277_1, arg278_1, buf352, 1568, 768, grid=grid(1568), stream=stream0)
        del arg277_1
        del arg278_1
        buf353 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [v_173], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (6144, 196), (196, 1), 0), reinterpret_tensor(arg279_1, (196, 196), (1, 196), 0), out=buf353)
        del arg279_1
        buf354 = reinterpret_tensor(buf352, (8, 196, 768), (150528, 768, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [x_410], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf348, arg276_1, buf353, arg280_1, buf354, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg276_1
        del arg280_1
        buf355 = reinterpret_tensor(buf347, (1568, 256), (256, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (1568, 768), (768, 1), 0), reinterpret_tensor(arg281_1, (768, 256), (1, 768), 0), out=buf355)
        del arg281_1
        buf356 = reinterpret_tensor(buf355, (8, 196, 256), (50176, 256, 1), 0); del buf355  # reuse
        buf360 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [x_406, x_413, layer_norm_117], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf356, buf331, buf343, arg272_1, arg282_1, arg283_1, arg284_1, buf360, 1568, 256, grid=grid(1568), stream=stream0)
        del arg272_1
        del arg282_1
        del arg283_1
        del arg284_1
        del buf331
        buf361 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf360, (1568, 256), (256, 1), 0), reinterpret_tensor(arg285_1, (256, 1536), (1, 256), 0), out=buf361)
        del arg285_1
        buf365 = reinterpret_tensor(buf354, (8, 768, 196), (150528, 196, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [v_175, v_176], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf361, arg286_1, arg287_1, arg288_1, buf365, 1568, 768, grid=grid(1568), stream=stream0)
        del arg287_1
        del arg288_1
        buf366 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [v_176], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (6144, 196), (196, 1), 0), reinterpret_tensor(arg289_1, (196, 196), (1, 196), 0), out=buf366)
        del arg289_1
        buf367 = reinterpret_tensor(buf365, (8, 196, 768), (150528, 768, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [x_417], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf361, arg286_1, buf366, arg290_1, buf367, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg286_1
        del arg290_1
        buf368 = reinterpret_tensor(buf360, (1568, 256), (256, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf367, (1568, 768), (768, 1), 0), reinterpret_tensor(arg291_1, (768, 256), (1, 768), 0), out=buf368)
        del arg291_1
        buf372 = reinterpret_tensor(buf343, (8, 196, 256), (50176, 256, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [x_420, layer_norm_119], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf356, buf368, arg292_1, arg293_1, arg294_1, buf372, 1568, 256, grid=grid(1568), stream=stream0)
        del arg293_1
        del arg294_1
        buf373 = buf361; del buf361  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf372, (1568, 256), (256, 1), 0), reinterpret_tensor(arg295_1, (256, 1536), (1, 256), 0), out=buf373)
        del arg295_1
        buf377 = reinterpret_tensor(buf367, (8, 768, 196), (150528, 196, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [v_178, v_179], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_3.run(buf373, arg296_1, arg297_1, arg298_1, buf377, 1568, 768, grid=grid(1568), stream=stream0)
        del arg297_1
        del arg298_1
        buf378 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [v_179], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf377, (6144, 196), (196, 1), 0), reinterpret_tensor(arg299_1, (196, 196), (1, 196), 0), out=buf378)
        del arg299_1
        buf379 = reinterpret_tensor(buf377, (8, 196, 768), (150528, 768, 1), 0); del buf377  # reuse
        # Topologically Sorted Source Nodes: [x_424], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf373, arg296_1, buf378, arg300_1, buf379, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg296_1
        del arg300_1
        del buf373
        del buf378
        buf380 = reinterpret_tensor(buf372, (1568, 256), (256, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf379, (1568, 768), (768, 1), 0), reinterpret_tensor(arg301_1, (768, 256), (1, 768), 0), out=buf380)
        del arg301_1
        del buf379
        buf381 = reinterpret_tensor(buf380, (8, 196, 256), (50176, 256, 1), 0); del buf380  # reuse
        buf382 = buf20; del buf20  # reuse
        buf383 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_420, x_427, x_428], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf381, buf356, buf368, arg292_1, arg302_1, buf382, buf383, 1568, 256, grid=grid(1568), stream=stream0)
        del arg292_1
        del arg302_1
        del buf356
        del buf368
        buf385 = empty_strided_cuda((8, 256, 2), (512, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_428, x_429], Original ATen: [aten.native_layer_norm, aten.mean]
        triton_red_fused_mean_native_layer_norm_12.run(buf381, buf382, buf383, arg303_1, arg304_1, buf385, 4096, 98, grid=grid(4096), stream=stream0)
        del arg303_1
        del arg304_1
        del buf381
        del buf382
        del buf383
        buf387 = empty_strided_cuda((8, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_428, x_429], Original ATen: [aten.native_layer_norm, aten.mean]
        triton_per_fused_mean_native_layer_norm_13.run(buf385, buf387, 2048, 2, grid=grid(2048), stream=stream0)
        del buf385
        buf388 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_428, x_429, x_431], Original ATen: [aten.native_layer_norm, aten.mean, aten.addmm]
        extern_kernels.addmm(arg306_1, buf387, reinterpret_tensor(arg305_1, (256, 1000), (1, 256), 0), alpha=1, beta=1, out=buf388)
        del arg305_1
        del arg306_1
        del buf387
    return (buf388, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((256, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1000, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gmlp_s16_224', benchmark_compiled_module)
