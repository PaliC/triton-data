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


# kernel path: /tmp/torchinductor_sahanp/rg/crg6vcs6v27zwe7cswxiifw3mfhbqmlnuw2nxsjqel43fb3nb5w3.py
# Topologically Sorted Source Nodes: [x_351, x_352], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_351 => add_177
#   x_352 => add_178, add_179, mul_222, mul_223, rsqrt_51, sub_75, var_mean_51
# Graph fragment:
#   %add_177 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_398, %arg3_1), kwargs = {})
#   %var_mean_51 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_177, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_177, %getitem_179), kwargs = {})
#   %add_178 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_178, 1e-06), kwargs = {})
#   %rsqrt_51 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_178,), kwargs = {})
#   %mul_222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_75, %rsqrt_51), kwargs = {})
#   %mul_223 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_222, %arg4_1), kwargs = {})
#   %add_179 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_223, %arg5_1), kwargs = {})
triton_red_fused_add_native_layer_norm_0 = async_compile.triton('triton_red_fused_add_native_layer_norm_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196) % 16
    x2 = (xindex // 3136)
    x4 = xindex % 3136
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(x1 % 4)) + (56*(x0 // 14)) + (784*(x1 // 4)) + (3136*r3) + (401408*x2) + (x0 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp9 = tl.load(in_ptr0 + ((14*(x1 % 4)) + (56*(x0 // 14)) + (784*(x1 // 4)) + (3136*r3) + (401408*x2) + (x0 % 14)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 128.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-06
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r3 + (128*x5)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/iu/ciuehhw7775v2ll7jcl4zhqjkl3yntkvw6ygd2xuzckb2cnbgljb.py
# Topologically Sorted Source Nodes: [x_353], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   x_353 => clone_174, mul_224
# Graph fragment:
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%getitem_180, 0.42044820762685725), kwargs = {})
#   %clone_174 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_96,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_1 = async_compile.triton('triton_poi_fused_clone_mul_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 3136
    x2 = (xindex // 100352) % 4
    x3 = (xindex // 401408)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (384*x1) + (1204224*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.42044820762685725
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ql/cqlmuvl7t7qfqn73rlmlvwbts72rhhex4txb277ngm6fkxl7br6d.py
# Topologically Sorted Source Nodes: [x_353], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   x_353 => clone_175, mul_225
# Graph fragment:
#   %mul_225 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_191, 0.42044820762685725), kwargs = {})
#   %clone_175 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_97,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_2 = async_compile.triton('triton_poi_fused_clone_mul_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32) % 16
    y2 = (yindex // 512) % 4
    y3 = (yindex // 2048)
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (128 + y0 + (32*y2) + (384*x4) + (75264*y1) + (1204224*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (128 + y0 + (32*y2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.42044820762685725
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4 + (196*y5)), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jx/cjx4r4cgvz7hq2rvohg55i6cb3qsqkwllj7qih6f45dag2dvdoje.py
# Topologically Sorted Source Nodes: [x_353], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   x_353 => amax_24, any_25, div_24, eq_24, exp_24, full_default, logical_not_48, logical_not_49, sub_76, sum_25, where_24
# Graph fragment:
#   %eq_24 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%view_404, -inf), kwargs = {})
#   %logical_not_48 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_24,), kwargs = {})
#   %any_25 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_48, -1, True), kwargs = {})
#   %logical_not_49 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_25,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 4, 16, 196, 196], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %amax_24 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_404, [-1], True), kwargs = {})
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_404, %amax_24), kwargs = {})
#   %exp_24 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_76,), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [-1], True), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_24, %sum_25), kwargs = {})
#   %where_24 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_49, %full_default, %div_24), kwargs = {})
triton_red_fused__safe_softmax_3 = async_compile.triton('triton_red_fused__safe_softmax_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__safe_softmax_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__safe_softmax_3(in_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.int1)
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    _tmp10 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = float("-inf")
        tmp2 = tmp0 == tmp1
        tmp3 = tmp2 == 0
        tmp4 = tmp3.to(tl.int64)
        tmp5 = (tmp4 != 0)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 | tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp9 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp11 = triton_helpers.maximum(_tmp10, tmp9)
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp7 = triton_helpers.any(_tmp7.to(tl.int8), 1)[:, None].to(tl.int1)
    tmp10 = triton_helpers.max2(_tmp10, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp12 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp12 - tmp10
        tmp14 = tl_math.exp(tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp19 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tmp7 == 0
        tmp20 = tmp19 - tmp10
        tmp21 = tl_math.exp(tmp20)
        tmp22 = tmp21 / tmp16
        tmp23 = 0.0
        tmp24 = tl.where(tmp18, tmp23, tmp22)
        tl.store(out_ptr3 + (r2 + (196*x3)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lq/clqidpn64idfwafeoqaifsiy26x3v72mgfvdhhvg4ouk6l7cqlv2.py
# Topologically Sorted Source Nodes: [x_353], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_353 => clone_176
# Graph fragment:
#   %clone_176 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_99,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 3136
    x2 = (xindex // 100352) % 4
    x3 = (xindex // 401408)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (32*x2) + (384*x1) + (1204224*x3)), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gc/cgchz255kyjl4lrh4363a37apa2elntzqxcit2xevs7rb4fyjtvd.py
# Topologically Sorted Source Nodes: [x_354], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_354 => clone_177
# Graph fragment:
#   %clone_177 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_192,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1048576, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 802816
    xnumel = 4
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 100352
    y1 = (yindex // 100352)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (100352*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (4*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/df/cdfqzerwlfxlj5q634vudvp5vpjd7jmgrffahwcc7jf5ssy3veaa.py
# Topologically Sorted Source Nodes: [x_351, x_357, x_358], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_351 => add_177
#   x_357 => add_180
#   x_358 => add_181, add_182, mul_226, mul_227, rsqrt_52, sub_77, var_mean_52
# Graph fragment:
#   %add_177 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_398, %arg3_1), kwargs = {})
#   %add_180 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_177, %view_410), kwargs = {})
#   %var_mean_52 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_180, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_180, %getitem_184), kwargs = {})
#   %add_181 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_183, 1e-06), kwargs = {})
#   %rsqrt_52 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_181,), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %rsqrt_52), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_226, %arg10_1), kwargs = {})
#   %add_182 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_227, %arg11_1), kwargs = {})
triton_red_fused_add_native_layer_norm_6 = async_compile.triton('triton_red_fused_add_native_layer_norm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196) % 16
    x2 = (xindex // 3136)
    x5 = xindex % 3136
    x4 = xindex
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(x1 % 4)) + (56*(x0 // 14)) + (784*(x1 // 4)) + (3136*r3) + (401408*x2) + (x0 % 14)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_out_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(in_out_ptr0 + (r3 + (128*x4)), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp13 - tmp10
        tmp15 = 128.0
        tmp16 = tmp11 / tmp15
        tmp17 = 1e-06
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r3 + (128*x4)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/to/ctote4rjghu4tkt7urrxkyqgu6kijj4fjzk4hafu3dqk4kddvo4p.py
# Topologically Sorted Source Nodes: [x_360], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_360 => add_183, erf_24, mul_228, mul_229, mul_230
# Graph fragment:
#   %mul_228 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_412, 0.5), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_412, 0.7071067811865476), kwargs = {})
#   %erf_24 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_229,), kwargs = {})
#   %add_183 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_24, 1), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_228, %add_183), kwargs = {})
triton_poi_fused_gelu_7 = async_compile.triton('triton_poi_fused_gelu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
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


# kernel path: /tmp/torchinductor_sahanp/u2/cu2pqzju5c2yztcdb7yjmcugqxjxff2acu3pjrp5ao65noptk23p.py
# Topologically Sorted Source Nodes: [x_364, x_365], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_364 => add_184
#   x_365 => add_185, add_186, mul_231, mul_232, rsqrt_53, sub_78, var_mean_53
# Graph fragment:
#   %add_184 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_180, %view_414), kwargs = {})
#   %var_mean_53 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_184, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_184, %getitem_186), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_185, 1e-06), kwargs = {})
#   %rsqrt_53 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_185,), kwargs = {})
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %rsqrt_53), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_231, %arg16_1), kwargs = {})
#   %add_186 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_232, %arg17_1), kwargs = {})
triton_per_fused_add_native_layer_norm_8 = async_compile.triton('triton_per_fused_add_native_layer_norm_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pq/cpq7xrtv63nkbcvhifomloqcgyzqbbq7kni2btnwwk3tqaqdghfi.py
# Topologically Sorted Source Nodes: [x_364, x_370, x_371], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_364 => add_184
#   x_370 => add_187
#   x_371 => add_188, add_189, mul_235, mul_236, rsqrt_54, sub_80, var_mean_54
# Graph fragment:
#   %add_184 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_180, %view_414), kwargs = {})
#   %add_187 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_184, %view_426), kwargs = {})
#   %var_mean_54 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_187, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_187, %getitem_191), kwargs = {})
#   %add_188 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_190, 1e-06), kwargs = {})
#   %rsqrt_54 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_188,), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %rsqrt_54), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_235, %arg22_1), kwargs = {})
#   %add_189 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_236, %arg23_1), kwargs = {})
triton_per_fused_add_native_layer_norm_9 = async_compile.triton('triton_per_fused_add_native_layer_norm_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 128.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp8, xmask)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp35, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/w5/cw52m7kxl4ddjz4bjcp55okpaii46fv2v2ybrvra7y6rwhmffc5n.py
# Topologically Sorted Source Nodes: [x_380], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_380 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_204, %arg28_1, %arg29_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_10 = async_compile.triton('triton_poi_fused_convolution_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128) % 56
    x2 = (xindex // 7168) % 56
    x3 = (xindex // 401408)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*(x1 % 14)) + (1792*(x2 % 14)) + (25088*(x1 // 14)) + (100352*(x2 // 14)) + (401408*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*(x1 % 14)) + (1792*(x2 % 14)) + (25088*(x1 // 14)) + (100352*(x2 // 14)) + (401408*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tu/ctu6hm3nuzwtjsmjeq6rse2hzvt4b46zdwgwbadyptyvasrxbhkg.py
# Topologically Sorted Source Nodes: [x_380], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_380 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_204, %arg28_1, %arg29_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_11 = async_compile.triton('triton_poi_fused_convolution_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
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


# kernel path: /tmp/torchinductor_sahanp/ky/ckyvo25eikmmpwbu6ofibyirebzsknri3wctf4el336tlqfnx6nt.py
# Topologically Sorted Source Nodes: [x_381], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_381 => var_mean_55
# Graph fragment:
#   %var_mean_55 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_205, [3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_12 = async_compile.triton('triton_per_fused_native_layer_norm_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_12(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 25088
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 256, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hl/chlcyaax3juq3jc33pwdnjf235uswkf3uzzum6423uqvm3kec4ta.py
# Topologically Sorted Source Nodes: [x_383], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_383 => constant_pad_nd_2
# Graph fragment:
#   %constant_pad_nd_2 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_206, [0, 1, 0, 1], -inf), kwargs = {})
triton_poi_fused_constant_pad_nd_13 = async_compile.triton('triton_poi_fused_constant_pad_nd_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6653952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 14592) % 57
    x1 = (xindex // 256) % 57
    x3 = (xindex // 831744)
    x4 = xindex % 14592
    x0 = xindex % 256
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 56, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (14336*x2) + (802816*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x1 + (56*x2) + (3136*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (x1 + (56*x2) + (3136*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 256.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp10 * tmp16
    tmp18 = tl.load(in_ptr4 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr5 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, float("-inf"), tmp21.dtype)
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tl.store(out_ptr0 + (x5), tmp23, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4t/c4tmd4xazc2mpq5etq2d4c5p7fvzh7xkhsjffgnkrasrqyni4edz.py
# Topologically Sorted Source Nodes: [x_383, x_384], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_383 => constant_pad_nd_2
#   x_384 => _low_memory_max_pool2d_with_offsets_2
# Graph fragment:
#   %constant_pad_nd_2 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_206, [0, 1, 0, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_2, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_14 = async_compile.triton('triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28) % 28
    y2 = (yindex // 784)
    y4 = yindex % 784
    tmp0 = tl.load(in_ptr0 + (x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (256 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (512 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (14592 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (14848 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (15104 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (29184 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (29440 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (29696 + x3 + (512*y0) + (29184*y1) + (831744*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y4 + (784*x3) + (200704*y2)), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6u/c6urpxhwwnzbgtyebq5crjxmopmxclzezuswsedzkmuzgzkcnrqf.py
# Topologically Sorted Source Nodes: [x_388, x_389], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_388 => add_194
#   x_389 => var_mean_56
# Graph fragment:
#   %add_194 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_434, %arg32_1), kwargs = {})
#   %var_mean_56 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_194, [3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_15 = async_compile.triton('triton_red_fused_add_native_layer_norm_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_15(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 196
    x2 = (xindex // 392) % 4
    x3 = (xindex // 1568)
    x5 = xindex % 1568
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(x2 % 2)) + (28*(x1 // 14)) + (392*(x2 // 2)) + (784*r4) + (100352*x0) + (200704*x3) + (x1 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r4 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x6), tmp4, xmask)
    tl.store(out_ptr1 + (x6), tmp5, xmask)
    tl.store(out_ptr2 + (x6), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/el/celdzcjxaojkhwkhtc4ndawprzlv3cnbkvpdsttfkgkt43tu4csu.py
# Topologically Sorted Source Nodes: [x_388, x_389], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_388 => add_194
#   x_389 => var_mean_56
# Graph fragment:
#   %add_194 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_434, %arg32_1), kwargs = {})
#   %var_mean_56 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_194, [3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_16 = async_compile.triton('triton_per_fused_add_native_layer_norm_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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


# kernel path: /tmp/torchinductor_sahanp/yp/cypozt5wacjn5rcggpza4u467xb7skawlmrg7xurjzduziwolzx5.py
# Topologically Sorted Source Nodes: [x_388, x_389], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_388 => add_194
#   x_389 => add_195, add_196, mul_242, mul_243, rsqrt_56, sub_82, var_mean_56
# Graph fragment:
#   %add_194 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_434, %arg32_1), kwargs = {})
#   %var_mean_56 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_194, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_194, %getitem_197), kwargs = {})
#   %add_195 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_196, 1e-06), kwargs = {})
#   %rsqrt_56 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_195,), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_82, %rsqrt_56), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_242, %arg33_1), kwargs = {})
#   %add_196 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_243, %arg34_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_17 = async_compile.triton('triton_poi_fused_add_native_layer_norm_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196) % 4
    y2 = (yindex // 784)
    y4 = yindex % 784
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + ((14*(y1 % 2)) + (28*(y0 // 14)) + (392*(y1 // 2)) + (784*x3) + (200704*y2) + (y0 % 14)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (256*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y5), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y5), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + (256*y5)), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ln/clns6mpkmhgbugraffj25mpaardk4fdxexuhwhm6guv5wmo7krg7.py
# Topologically Sorted Source Nodes: [x_390], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   x_390 => clone_190, mul_244
# Graph fragment:
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%getitem_198, 0.42044820762685725), kwargs = {})
#   %clone_190 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_104,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_18 = async_compile.triton('triton_poi_fused_clone_mul_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_18(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 784
    x2 = (xindex // 25088) % 8
    x3 = (xindex // 200704)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (768*x1) + (602112*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.42044820762685725
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qz/cqzi2ynb3pkf57lm7tec5boztmfbxgejbxft2jgijfwkn4amkmhe.py
# Topologically Sorted Source Nodes: [x_390], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   x_390 => clone_191, mul_245
# Graph fragment:
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_211, 0.42044820762685725), kwargs = {})
#   %clone_191 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_105,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_19 = async_compile.triton('triton_poi_fused_clone_mul_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_19(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32) % 4
    y2 = (yindex // 128) % 8
    y3 = (yindex // 1024)
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (256 + y0 + (32*y2) + (768*x4) + (150528*y1) + (602112*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (256 + y0 + (32*y2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.42044820762685725
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4 + (196*y5)), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gz/cgzzgn77rtgontdkxw2xjz7whhcj63mf3cbvf26fs5weneuljjmh.py
# Topologically Sorted Source Nodes: [x_390], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   x_390 => amax_26, any_27, div_26, eq_26, exp_26, full_default_2, logical_not_52, logical_not_53, sub_83, sum_27, where_26
# Graph fragment:
#   %eq_26 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%view_440, -inf), kwargs = {})
#   %logical_not_52 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_26,), kwargs = {})
#   %any_27 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_52, -1, True), kwargs = {})
#   %logical_not_53 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_27,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 8, 4, 196, 196], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %amax_26 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_440, [-1], True), kwargs = {})
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_440, %amax_26), kwargs = {})
#   %exp_26 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_83,), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_26, [-1], True), kwargs = {})
#   %div_26 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_26, %sum_27), kwargs = {})
#   %where_26 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_53, %full_default_2, %div_26), kwargs = {})
triton_per_fused__safe_softmax_20 = async_compile.triton('triton_per_fused__safe_softmax_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__safe_softmax_20(in_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = float("-inf")
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2 == 0
    tmp4 = tmp3.to(tl.int64)
    tmp5 = (tmp4 != 0)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.any(tmp8, 1)[:, None]
    tmp10 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, float("-inf"))
    tmp13 = triton_helpers.max2(tmp12, 1)[:, None]
    tmp14 = tmp0 - tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tmp9 == 0
    tmp21 = tmp15 / tmp19
    tmp22 = 0.0
    tmp23 = tl.where(tmp20, tmp22, tmp21)
    tl.store(out_ptr3 + (r1 + (196*x0)), tmp23, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h4/ch4krxaadqxzm4ye62yz325qqqd7a5ohzdhn7gjzhdv232w42gss.py
# Topologically Sorted Source Nodes: [x_390], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_390 => clone_192
# Graph fragment:
#   %clone_192 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_107,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_21 = async_compile.triton('triton_poi_fused_clone_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 784
    x2 = (xindex // 25088) % 8
    x3 = (xindex // 200704)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (32*x2) + (768*x1) + (602112*x3)), None)
    tmp1 = tl.load(in_ptr1 + (512 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bw/cbwn3tf674c4hznb5oiegxvyxugm5pkxhmamsjhxze63zzyp5tlb.py
# Topologically Sorted Source Nodes: [x_391], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_391 => clone_193
# Graph fragment:
#   %clone_193 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_212,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_22 = async_compile.triton('triton_poi_fused_clone_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[262144, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_22(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 200704
    xnumel = 8
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 25088
    y1 = (yindex // 25088)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (25088*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (8*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/s5/cs57wdfsjcygydrfcsupwhdgiyyupjrhdx2ar5cbjk6nvntqidti.py
# Topologically Sorted Source Nodes: [x_388, x_394, x_395], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_388 => add_194
#   x_394 => add_197
#   x_395 => var_mean_57
# Graph fragment:
#   %add_194 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_434, %arg32_1), kwargs = {})
#   %add_197 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_194, %view_446), kwargs = {})
#   %var_mean_57 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_197, [3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_23 = async_compile.triton('triton_red_fused_add_native_layer_norm_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 196
    x2 = (xindex // 392) % 4
    x3 = (xindex // 1568)
    x5 = xindex % 1568
    x6 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(x2 % 2)) + (28*(x1 // 14)) + (392*(x2 // 2)) + (784*r4) + (100352*x0) + (200704*x3) + (x1 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r4 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r4 + (128*x6)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r4 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x6), tmp8, xmask)
    tl.store(out_ptr1 + (x6), tmp9, xmask)
    tl.store(out_ptr2 + (x6), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/m7/cm7jymci57qdxje3w7qqi6ugjapnxt65ymbzrqn4x3qvbm4dfhix.py
# Topologically Sorted Source Nodes: [x_388, x_394, x_395], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_388 => add_194
#   x_394 => add_197
#   x_395 => add_198, add_199, mul_246, mul_247, rsqrt_57, sub_84, var_mean_57
# Graph fragment:
#   %add_194 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_434, %arg32_1), kwargs = {})
#   %add_197 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_194, %view_446), kwargs = {})
#   %var_mean_57 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_197, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_197, %getitem_202), kwargs = {})
#   %add_198 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_201, 1e-06), kwargs = {})
#   %rsqrt_57 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_198,), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %rsqrt_57), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_246, %arg39_1), kwargs = {})
#   %add_199 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_247, %arg40_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_24 = async_compile.triton('triton_poi_fused_add_native_layer_norm_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196) % 4
    y2 = (yindex // 784)
    y4 = yindex % 784
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + ((14*(y1 % 2)) + (28*(y0 // 14)) + (392*(y1 // 2)) + (784*x3) + (200704*y2) + (y0 % 14)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (256*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3 + (256*y5)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y5), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y5), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + (256*y5)), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kl/cklhja7kbziezf7lfwzzi64vjoxc7kbl26obyrns3msfk43qtrme.py
# Topologically Sorted Source Nodes: [x_397], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_397 => add_200, erf_26, mul_248, mul_249, mul_250
# Graph fragment:
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_448, 0.5), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_448, 0.7071067811865476), kwargs = {})
#   %erf_26 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_249,), kwargs = {})
#   %add_200 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_26, 1), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_248, %add_200), kwargs = {})
triton_poi_fused_gelu_25 = async_compile.triton('triton_poi_fused_gelu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_25(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_sahanp/jq/cjqnfortrrpkhsbebvd5hoz6uenfjhq6bwzk43j5thq3vcwxc5fp.py
# Topologically Sorted Source Nodes: [x_388, x_394, x_401, x_402], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_388 => add_194
#   x_394 => add_197
#   x_401 => add_201
#   x_402 => add_202, add_203, mul_251, mul_252, rsqrt_58, sub_85, var_mean_58
# Graph fragment:
#   %add_194 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_434, %arg32_1), kwargs = {})
#   %add_197 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_194, %view_446), kwargs = {})
#   %add_201 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_197, %view_450), kwargs = {})
#   %var_mean_58 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_201, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_201, %getitem_204), kwargs = {})
#   %add_202 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_203, 1e-06), kwargs = {})
#   %rsqrt_58 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_202,), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %rsqrt_58), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_251, %arg45_1), kwargs = {})
#   %add_203 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_252, %arg46_1), kwargs = {})
triton_red_fused_add_native_layer_norm_26 = async_compile.triton('triton_red_fused_add_native_layer_norm_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196) % 4
    x2 = (xindex // 784)
    x5 = xindex % 784
    x4 = xindex
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(x1 % 2)) + (28*(x0 // 14)) + (392*(x1 // 2)) + (784*r3) + (200704*x2) + (x0 % 14)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (256*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_out_ptr0 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
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
        tl.store(in_out_ptr0 + (r3 + (256*x4)), tmp10, rmask & xmask)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr6 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp15 - tmp12
        tmp17 = 256.0
        tmp18 = tmp13 / tmp17
        tmp19 = 1e-06
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp24 = tmp22 * tmp23
        tmp26 = tmp24 + tmp25
        tl.store(out_ptr2 + (r3 + (256*x4)), tmp26, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pg/cpghlc26tf3yxhzcit6plui7fslzb2uv5utuxtuxj64spvhhj72s.py
# Topologically Sorted Source Nodes: [x_407, x_408], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_407 => add_204
#   x_408 => add_205, add_206, mul_255, mul_256, rsqrt_59, sub_87, var_mean_59
# Graph fragment:
#   %add_204 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_201, %view_462), kwargs = {})
#   %var_mean_59 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_204, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_204, %getitem_209), kwargs = {})
#   %add_205 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_208, 1e-06), kwargs = {})
#   %rsqrt_59 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_205,), kwargs = {})
#   %mul_255 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %rsqrt_59), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_255, %arg51_1), kwargs = {})
#   %add_206 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_256, %arg52_1), kwargs = {})
triton_per_fused_add_native_layer_norm_27 = async_compile.triton('triton_per_fused_add_native_layer_norm_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 6272
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


# kernel path: /tmp/torchinductor_sahanp/ro/cron2a6wrvp6j4ze7h3kuq4zifhpazykbelz35ef36jfmcdqypjq.py
# Topologically Sorted Source Nodes: [x_417], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_417 => convolution_5
# Graph fragment:
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_224, %arg57_1, %arg58_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_28 = async_compile.triton('triton_poi_fused_convolution_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256) % 28
    x2 = (xindex // 7168) % 28
    x3 = (xindex // 200704)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*(x1 % 14)) + (3584*(x2 % 14)) + (50176*(x1 // 14)) + (100352*(x2 // 14)) + (200704*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*(x1 % 14)) + (3584*(x2 % 14)) + (50176*(x1 // 14)) + (100352*(x2 // 14)) + (200704*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (256*(x1 % 14)) + (3584*(x2 % 14)) + (50176*(x1 // 14)) + (100352*(x2 // 14)) + (200704*x3)), None)
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (x4), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6p/c6pkrjid5abwi4q6aazn27cttlmelsnbabcapcjmpex27b2yslz2.py
# Topologically Sorted Source Nodes: [x_417], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_417 => convolution_5
# Graph fragment:
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_224, %arg57_1, %arg58_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_29 = async_compile.triton('triton_poi_fused_convolution_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
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


# kernel path: /tmp/torchinductor_sahanp/od/cod24qdljziks5lg74pichmdkm22bg4wqfqqtcmulaqyimudi5qp.py
# Topologically Sorted Source Nodes: [x_418], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_418 => var_mean_60
# Graph fragment:
#   %var_mean_60 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_225, [3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_30 = async_compile.triton('triton_per_fused_native_layer_norm_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_30(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 6272
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/se/cseh7lor5eny7meml53nvpmh75gddxg7qf6w5hnoh2ywnurigfpu.py
# Topologically Sorted Source Nodes: [x_420], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_420 => constant_pad_nd_3
# Graph fragment:
#   %constant_pad_nd_3 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_226, [0, 1, 0, 1], -inf), kwargs = {})
triton_poi_fused_constant_pad_nd_31 = async_compile.triton('triton_poi_fused_constant_pad_nd_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3444736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 14848) % 29
    x1 = (xindex // 512) % 29
    x3 = (xindex // 430592)
    x4 = xindex % 14848
    x0 = xindex % 512
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 28, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (14336*x2) + (401408*x3)), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x1 + (28*x2) + (784*x3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (x1 + (28*x2) + (784*x3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp12 = 512.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp10 * tmp16
    tmp18 = tl.load(in_ptr4 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr5 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, float("-inf"), tmp21.dtype)
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tl.store(out_ptr0 + (x5), tmp23, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ei/cei4covmf36kcp3rhmxnio42npnylvt43p23sloragcjks4qhitu.py
# Topologically Sorted Source Nodes: [x_420, x_421], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_420 => constant_pad_nd_3
#   x_421 => _low_memory_max_pool2d_with_offsets_3
# Graph fragment:
#   %constant_pad_nd_3 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_226, [0, 1, 0, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_3, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_32 = async_compile.triton('triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 14
    y1 = (yindex // 14) % 14
    y2 = (yindex // 196)
    y4 = yindex % 196
    tmp0 = tl.load(in_ptr0 + (x3 + (1024*y0) + (29696*y1) + (430592*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (512 + x3 + (1024*y0) + (29696*y1) + (430592*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1024 + x3 + (1024*y0) + (29696*y1) + (430592*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (14848 + x3 + (1024*y0) + (29696*y1) + (430592*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (15360 + x3 + (1024*y0) + (29696*y1) + (430592*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (15872 + x3 + (1024*y0) + (29696*y1) + (430592*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (29696 + x3 + (1024*y0) + (29696*y1) + (430592*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (30208 + x3 + (1024*y0) + (29696*y1) + (430592*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (30720 + x3 + (1024*y0) + (29696*y1) + (430592*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y4 + (196*x3) + (100352*y2)), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7v/c7vk44sbylvtcnxk2tevxlpwmqnbf7x2jksppwfearejkano3w7c.py
# Topologically Sorted Source Nodes: [x_425, x_426], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_425 => add_211
#   x_426 => var_mean_61
# Graph fragment:
#   %add_211 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_470, %arg61_1), kwargs = {})
#   %var_mean_61 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_211, [3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_33 = async_compile.triton('triton_red_fused_add_native_layer_norm_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_33(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4) % 196
    x2 = (xindex // 784)
    x4 = xindex % 784
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (100352*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/at/catwcdhj4j24nz2zqxuawoislud4egnlgb4b7hd47asjboquf3rl.py
# Topologically Sorted Source Nodes: [x_425, x_426], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_425 => add_211
#   x_426 => var_mean_61
# Graph fragment:
#   %add_211 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_470, %arg61_1), kwargs = {})
#   %var_mean_61 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_211, [3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_34 = async_compile.triton('triton_per_fused_add_native_layer_norm_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_34(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (4*x0)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (4*x0)), xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/v6/cv6qpu22cnm7pbqe2sz7smzgvk4jzf7hpzkeu4qppo4p27lelen7.py
# Topologically Sorted Source Nodes: [x_425, x_426], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_425 => add_211
#   x_426 => add_212, add_213, mul_262, mul_263, rsqrt_61, sub_89, var_mean_61
# Graph fragment:
#   %add_211 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_470, %arg61_1), kwargs = {})
#   %var_mean_61 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_211, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_211, %getitem_215), kwargs = {})
#   %add_212 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_214, 1e-06), kwargs = {})
#   %rsqrt_61 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_212,), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %rsqrt_61), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %arg62_1), kwargs = {})
#   %add_213 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %arg63_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_35 = async_compile.triton('triton_poi_fused_add_native_layer_norm_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 512.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7o/c7o33rzliazbwujisqhtyfcqdya6wdgyczeewlgxokf5sf2cheqo.py
# Topologically Sorted Source Nodes: [x_427], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   x_427 => clone_205, mul_264
# Graph fragment:
#   %mul_264 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%getitem_216, 0.42044820762685725), kwargs = {})
#   %clone_205 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_112,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_36 = async_compile.triton('triton_poi_fused_clone_mul_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_36(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 196
    x2 = (xindex // 6272) % 16
    x3 = (xindex // 100352)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1536*x1) + (301056*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.42044820762685725
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/j5/cj5kdjwbbovqsmuughchukmicuna27zszvel2d722ijkl2jdrvbu.py
# Topologically Sorted Source Nodes: [x_427], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   x_427 => clone_206, mul_265
# Graph fragment:
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_231, 0.42044820762685725), kwargs = {})
#   %clone_206 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_113,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_37 = async_compile.triton('triton_poi_fused_clone_mul_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_37(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (512 + y0 + (1536*x2) + (301056*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (512 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.42044820762685725
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zh/czheetgaiddwtz4dbgkvg75b5vjczp6rxnkz6isqtsm56whjcafe.py
# Topologically Sorted Source Nodes: [x_427], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   x_427 => amax_28, any_29, div_28, eq_28, exp_28, full_default_4, logical_not_56, logical_not_57, sub_90, sum_29, where_28
# Graph fragment:
#   %eq_28 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%view_476, -inf), kwargs = {})
#   %logical_not_56 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_28,), kwargs = {})
#   %any_29 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_56, -1, True), kwargs = {})
#   %logical_not_57 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_29,), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 16, 1, 196, 196], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %amax_28 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_476, [-1], True), kwargs = {})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_476, %amax_28), kwargs = {})
#   %exp_28 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_90,), kwargs = {})
#   %sum_29 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_28, [-1], True), kwargs = {})
#   %div_28 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_28, %sum_29), kwargs = {})
#   %where_28 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_57, %full_default_4, %div_28), kwargs = {})
triton_red_fused__safe_softmax_38 = async_compile.triton('triton_red_fused__safe_softmax_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__safe_softmax_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__safe_softmax_38(in_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.int1)
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    _tmp10 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = float("-inf")
        tmp2 = tmp0 == tmp1
        tmp3 = tmp2 == 0
        tmp4 = tmp3.to(tl.int64)
        tmp5 = (tmp4 != 0)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 | tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp9 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp11 = triton_helpers.maximum(_tmp10, tmp9)
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp7 = triton_helpers.any(_tmp7.to(tl.int8), 1)[:, None].to(tl.int1)
    tmp10 = triton_helpers.max2(_tmp10, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp12 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp12 - tmp10
        tmp14 = tl_math.exp(tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    x4 = xindex % 196
    x6 = (xindex // 196)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp19 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tmp7 == 0
        tmp20 = tmp19 - tmp10
        tmp21 = tl_math.exp(tmp20)
        tmp22 = tmp21 / tmp16
        tmp23 = 0.0
        tmp24 = tl.where(tmp18, tmp23, tmp22)
        tl.store(out_ptr3 + (r2 + (196*x4) + (38432*x6)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zn/cznxyi2minm5pz73hirmc6u66ivwu3yzfmq64q7w75okcwtkp5xo.py
# Topologically Sorted Source Nodes: [x_427], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_427 => clone_207
# Graph fragment:
#   %clone_207 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_115,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_39 = async_compile.triton('triton_poi_fused_clone_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_39(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 196
    x2 = (xindex // 6272) % 16
    x3 = (xindex // 100352)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (32*x2) + (1536*x1) + (301056*x3)), None)
    tmp1 = tl.load(in_ptr1 + (1024 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/r3/cr352uojlsrncaezhojk6wfvzfaqqdzqunkxvaauwz35h3djqeo4.py
# Topologically Sorted Source Nodes: [x_428], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_428 => clone_208
# Graph fragment:
#   %clone_208 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_232,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_40 = async_compile.triton('triton_poi_fused_clone_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_40(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 6272
    y1 = (yindex // 6272)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (6272*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kv/ckvtsr572a4xzincpr6vv6g7wn7gyojjeh7w4z6s5ztegjjwp7um.py
# Topologically Sorted Source Nodes: [x_425, x_431, x_432], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_425 => add_211
#   x_431 => add_214
#   x_432 => var_mean_62
# Graph fragment:
#   %add_211 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_470, %arg61_1), kwargs = {})
#   %add_214 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_211, %view_482), kwargs = {})
#   %var_mean_62 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_214, [3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_41 = async_compile.triton('triton_red_fused_add_native_layer_norm_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4) % 196
    x2 = (xindex // 784)
    x4 = xindex % 784
    x5 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (100352*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/ln/clnzztdp34dyuwepc626sl2mqplpx2xgcduaiyw3er4fbhk7nai6.py
# Topologically Sorted Source Nodes: [x_425, x_431, x_432], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_425 => add_211
#   x_431 => add_214
#   x_432 => add_215, add_216, mul_266, mul_267, rsqrt_62, sub_91, var_mean_62
# Graph fragment:
#   %add_211 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_470, %arg61_1), kwargs = {})
#   %add_214 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_211, %view_482), kwargs = {})
#   %var_mean_62 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_214, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_214, %getitem_220), kwargs = {})
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_219, 1e-06), kwargs = {})
#   %rsqrt_62 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_215,), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %rsqrt_62), kwargs = {})
#   %mul_267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_266, %arg68_1), kwargs = {})
#   %add_216 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_267, %arg69_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_42 = async_compile.triton('triton_poi_fused_add_native_layer_norm_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 512.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pi/cpie7pwvtizr62j4ranq4xd5alqemyiwm3yakvfvyiy75afayswn.py
# Topologically Sorted Source Nodes: [x_434], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_434 => add_217, erf_28, mul_268, mul_269, mul_270
# Graph fragment:
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_484, 0.5), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_484, 0.7071067811865476), kwargs = {})
#   %erf_28 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_269,), kwargs = {})
#   %add_217 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_28, 1), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_268, %add_217), kwargs = {})
triton_poi_fused_gelu_43 = async_compile.triton('triton_poi_fused_gelu_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_43(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_sahanp/hn/chnhueddkmjtzsrh4k7ry75sbjv3jmilbzmzmjsonzjblcanhtbp.py
# Topologically Sorted Source Nodes: [x_425, x_431, x_438, x_439], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_425 => add_211
#   x_431 => add_214
#   x_438 => add_218
#   x_439 => add_219, add_220, mul_271, mul_272, rsqrt_63, sub_92, var_mean_63
# Graph fragment:
#   %add_211 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_470, %arg61_1), kwargs = {})
#   %add_214 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_211, %view_482), kwargs = {})
#   %add_218 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_214, %view_486), kwargs = {})
#   %var_mean_63 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_218, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_218, %getitem_222), kwargs = {})
#   %add_219 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_221, 1e-06), kwargs = {})
#   %rsqrt_63 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_219,), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %rsqrt_63), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %arg74_1), kwargs = {})
#   %add_220 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_272, %arg75_1), kwargs = {})
triton_red_fused_add_native_layer_norm_44 = async_compile.triton('triton_red_fused_add_native_layer_norm_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (512*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (r2 + (512*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp10, rmask & xmask)
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
        tmp15 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp15 - tmp12
        tmp17 = 512.0
        tmp18 = tmp13 / tmp17
        tmp19 = 1e-06
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp24 = tmp22 * tmp23
        tmp26 = tmp24 + tmp25
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp26, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ax/caxssyhobfwixcn6tzkyt7urbztzw3jj5zvvhv37abyexmlxdnee.py
# Topologically Sorted Source Nodes: [x_444, x_445], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_444 => add_221
#   x_445 => add_222, add_223, mul_275, mul_276, rsqrt_64, sub_94, var_mean_64
# Graph fragment:
#   %add_221 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_218, %view_498), kwargs = {})
#   %var_mean_64 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_221, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_221, %getitem_227), kwargs = {})
#   %add_222 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_226, 1e-06), kwargs = {})
#   %rsqrt_64 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_222,), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %rsqrt_64), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_275, %arg80_1), kwargs = {})
#   %add_223 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_276, %arg81_1), kwargs = {})
triton_per_fused_add_native_layer_norm_45 = async_compile.triton('triton_per_fused_add_native_layer_norm_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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


# kernel path: /tmp/torchinductor_sahanp/qm/cqmvhyyxurcvupudpahpttytab5jtkd3bbvdnniwzqo5l6lpn5lb.py
# Topologically Sorted Source Nodes: [x_444, x_451, x_452], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_444 => add_221
#   x_451 => add_225
#   x_452 => add_226, add_227, mul_280, mul_281, rsqrt_65, sub_95, var_mean_65
# Graph fragment:
#   %add_221 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_218, %view_498), kwargs = {})
#   %add_225 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_221, %view_502), kwargs = {})
#   %var_mean_65 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_225, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_225, %getitem_229), kwargs = {})
#   %add_226 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_228, 1e-06), kwargs = {})
#   %rsqrt_65 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_226,), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %rsqrt_65), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_280, %arg86_1), kwargs = {})
#   %add_227 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_281, %arg87_1), kwargs = {})
triton_per_fused_add_native_layer_norm_46 = async_compile.triton('triton_per_fused_add_native_layer_norm_46', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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


# kernel path: /tmp/torchinductor_sahanp/2o/c2owpkze5sodjpr6ygcop3rbrzvz6753rrbvmpwfbdp2eo3knjxl.py
# Topologically Sorted Source Nodes: [x_678, x_685, x_688], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_678 => add_347
#   x_685 => add_351
#   x_688 => var_mean_101
# Graph fragment:
#   %add_347 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_344, %view_786), kwargs = {})
#   %add_351 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_347, %view_790), kwargs = {})
#   %var_mean_101 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_371, [3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_47 = async_compile.triton('triton_per_fused_add_native_layer_norm_47', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1568
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
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, None)
    tl.store(out_ptr0 + (x0), tmp16, None)
    tl.store(out_ptr1 + (x0), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rc/crcestzt2fptbwaubpqn5bt5ch64zzbt3brzwiktxjtb5qe3mh6t.py
# Topologically Sorted Source Nodes: [x_690], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_690 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%permute_372, [-1, -2], True), kwargs = {})
triton_red_fused_mean_48 = async_compile.triton('triton_red_fused_mean_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (98*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (98*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 512.0
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


# kernel path: /tmp/torchinductor_sahanp/yj/cyj36p43nc23msplus3j2xv4bvoj2fgyg5t52gwqhcelgdxlqrre.py
# Topologically Sorted Source Nodes: [x_690], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_690 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%permute_372, [-1, -2], True), kwargs = {})
triton_per_fused_mean_49 = async_compile.triton('triton_per_fused_mean_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_49(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (1024*x1)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 196.0
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr1 + (x3), tmp5, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (1, 16, 196, 128), (401408, 25088, 128, 1))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (384, 128), (128, 1))
    assert_size_stride(arg7_1, (384, ), (1, ))
    assert_size_stride(arg8_1, (128, 128), (128, 1))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (512, 128), (128, 1))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (128, 512), (512, 1))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, ), (1, ))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (384, 128), (128, 1))
    assert_size_stride(arg19_1, (384, ), (1, ))
    assert_size_stride(arg20_1, (128, 128), (128, 1))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (512, 128), (128, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (128, 512), (512, 1))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (1, 4, 196, 256), (200704, 50176, 256, 1))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (768, 256), (256, 1))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (256, 256), (256, 1))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (1024, 256), (256, 1))
    assert_size_stride(arg42_1, (1024, ), (1, ))
    assert_size_stride(arg43_1, (256, 1024), (1024, 1))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (768, 256), (256, 1))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (256, 256), (256, 1))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (1024, 256), (256, 1))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (256, 1024), (1024, 1))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (512, ), (1, ))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (1, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (512, ), (1, ))
    assert_size_stride(arg64_1, (1536, 512), (512, 1))
    assert_size_stride(arg65_1, (1536, ), (1, ))
    assert_size_stride(arg66_1, (512, 512), (512, 1))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (2048, 512), (512, 1))
    assert_size_stride(arg71_1, (2048, ), (1, ))
    assert_size_stride(arg72_1, (512, 2048), (2048, 1))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (1536, 512), (512, 1))
    assert_size_stride(arg77_1, (1536, ), (1, ))
    assert_size_stride(arg78_1, (512, 512), (512, 1))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (2048, 512), (512, 1))
    assert_size_stride(arg83_1, (2048, ), (1, ))
    assert_size_stride(arg84_1, (512, 2048), (2048, 1))
    assert_size_stride(arg85_1, (512, ), (1, ))
    assert_size_stride(arg86_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (1536, 512), (512, 1))
    assert_size_stride(arg89_1, (1536, ), (1, ))
    assert_size_stride(arg90_1, (512, 512), (512, 1))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (2048, 512), (512, 1))
    assert_size_stride(arg95_1, (2048, ), (1, ))
    assert_size_stride(arg96_1, (512, 2048), (2048, 1))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (1536, 512), (512, 1))
    assert_size_stride(arg101_1, (1536, ), (1, ))
    assert_size_stride(arg102_1, (512, 512), (512, 1))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (2048, 512), (512, 1))
    assert_size_stride(arg107_1, (2048, ), (1, ))
    assert_size_stride(arg108_1, (512, 2048), (2048, 1))
    assert_size_stride(arg109_1, (512, ), (1, ))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (512, ), (1, ))
    assert_size_stride(arg112_1, (1536, 512), (512, 1))
    assert_size_stride(arg113_1, (1536, ), (1, ))
    assert_size_stride(arg114_1, (512, 512), (512, 1))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (512, ), (1, ))
    assert_size_stride(arg117_1, (512, ), (1, ))
    assert_size_stride(arg118_1, (2048, 512), (512, 1))
    assert_size_stride(arg119_1, (2048, ), (1, ))
    assert_size_stride(arg120_1, (512, 2048), (2048, 1))
    assert_size_stride(arg121_1, (512, ), (1, ))
    assert_size_stride(arg122_1, (512, ), (1, ))
    assert_size_stride(arg123_1, (512, ), (1, ))
    assert_size_stride(arg124_1, (1536, 512), (512, 1))
    assert_size_stride(arg125_1, (1536, ), (1, ))
    assert_size_stride(arg126_1, (512, 512), (512, 1))
    assert_size_stride(arg127_1, (512, ), (1, ))
    assert_size_stride(arg128_1, (512, ), (1, ))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (2048, 512), (512, 1))
    assert_size_stride(arg131_1, (2048, ), (1, ))
    assert_size_stride(arg132_1, (512, 2048), (2048, 1))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (512, ), (1, ))
    assert_size_stride(arg135_1, (512, ), (1, ))
    assert_size_stride(arg136_1, (1536, 512), (512, 1))
    assert_size_stride(arg137_1, (1536, ), (1, ))
    assert_size_stride(arg138_1, (512, 512), (512, 1))
    assert_size_stride(arg139_1, (512, ), (1, ))
    assert_size_stride(arg140_1, (512, ), (1, ))
    assert_size_stride(arg141_1, (512, ), (1, ))
    assert_size_stride(arg142_1, (2048, 512), (512, 1))
    assert_size_stride(arg143_1, (2048, ), (1, ))
    assert_size_stride(arg144_1, (512, 2048), (2048, 1))
    assert_size_stride(arg145_1, (512, ), (1, ))
    assert_size_stride(arg146_1, (512, ), (1, ))
    assert_size_stride(arg147_1, (512, ), (1, ))
    assert_size_stride(arg148_1, (1536, 512), (512, 1))
    assert_size_stride(arg149_1, (1536, ), (1, ))
    assert_size_stride(arg150_1, (512, 512), (512, 1))
    assert_size_stride(arg151_1, (512, ), (1, ))
    assert_size_stride(arg152_1, (512, ), (1, ))
    assert_size_stride(arg153_1, (512, ), (1, ))
    assert_size_stride(arg154_1, (2048, 512), (512, 1))
    assert_size_stride(arg155_1, (2048, ), (1, ))
    assert_size_stride(arg156_1, (512, 2048), (2048, 1))
    assert_size_stride(arg157_1, (512, ), (1, ))
    assert_size_stride(arg158_1, (512, ), (1, ))
    assert_size_stride(arg159_1, (512, ), (1, ))
    assert_size_stride(arg160_1, (1536, 512), (512, 1))
    assert_size_stride(arg161_1, (1536, ), (1, ))
    assert_size_stride(arg162_1, (512, 512), (512, 1))
    assert_size_stride(arg163_1, (512, ), (1, ))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (2048, 512), (512, 1))
    assert_size_stride(arg167_1, (2048, ), (1, ))
    assert_size_stride(arg168_1, (512, 2048), (2048, 1))
    assert_size_stride(arg169_1, (512, ), (1, ))
    assert_size_stride(arg170_1, (512, ), (1, ))
    assert_size_stride(arg171_1, (512, ), (1, ))
    assert_size_stride(arg172_1, (1536, 512), (512, 1))
    assert_size_stride(arg173_1, (1536, ), (1, ))
    assert_size_stride(arg174_1, (512, 512), (512, 1))
    assert_size_stride(arg175_1, (512, ), (1, ))
    assert_size_stride(arg176_1, (512, ), (1, ))
    assert_size_stride(arg177_1, (512, ), (1, ))
    assert_size_stride(arg178_1, (2048, 512), (512, 1))
    assert_size_stride(arg179_1, (2048, ), (1, ))
    assert_size_stride(arg180_1, (512, 2048), (2048, 1))
    assert_size_stride(arg181_1, (512, ), (1, ))
    assert_size_stride(arg182_1, (512, ), (1, ))
    assert_size_stride(arg183_1, (512, ), (1, ))
    assert_size_stride(arg184_1, (1536, 512), (512, 1))
    assert_size_stride(arg185_1, (1536, ), (1, ))
    assert_size_stride(arg186_1, (512, 512), (512, 1))
    assert_size_stride(arg187_1, (512, ), (1, ))
    assert_size_stride(arg188_1, (512, ), (1, ))
    assert_size_stride(arg189_1, (512, ), (1, ))
    assert_size_stride(arg190_1, (2048, 512), (512, 1))
    assert_size_stride(arg191_1, (2048, ), (1, ))
    assert_size_stride(arg192_1, (512, 2048), (2048, 1))
    assert_size_stride(arg193_1, (512, ), (1, ))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (512, ), (1, ))
    assert_size_stride(arg196_1, (1536, 512), (512, 1))
    assert_size_stride(arg197_1, (1536, ), (1, ))
    assert_size_stride(arg198_1, (512, 512), (512, 1))
    assert_size_stride(arg199_1, (512, ), (1, ))
    assert_size_stride(arg200_1, (512, ), (1, ))
    assert_size_stride(arg201_1, (512, ), (1, ))
    assert_size_stride(arg202_1, (2048, 512), (512, 1))
    assert_size_stride(arg203_1, (2048, ), (1, ))
    assert_size_stride(arg204_1, (512, 2048), (2048, 1))
    assert_size_stride(arg205_1, (512, ), (1, ))
    assert_size_stride(arg206_1, (512, ), (1, ))
    assert_size_stride(arg207_1, (512, ), (1, ))
    assert_size_stride(arg208_1, (1536, 512), (512, 1))
    assert_size_stride(arg209_1, (1536, ), (1, ))
    assert_size_stride(arg210_1, (512, 512), (512, 1))
    assert_size_stride(arg211_1, (512, ), (1, ))
    assert_size_stride(arg212_1, (512, ), (1, ))
    assert_size_stride(arg213_1, (512, ), (1, ))
    assert_size_stride(arg214_1, (2048, 512), (512, 1))
    assert_size_stride(arg215_1, (2048, ), (1, ))
    assert_size_stride(arg216_1, (512, 2048), (2048, 1))
    assert_size_stride(arg217_1, (512, ), (1, ))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (512, ), (1, ))
    assert_size_stride(arg220_1, (1536, 512), (512, 1))
    assert_size_stride(arg221_1, (1536, ), (1, ))
    assert_size_stride(arg222_1, (512, 512), (512, 1))
    assert_size_stride(arg223_1, (512, ), (1, ))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (512, ), (1, ))
    assert_size_stride(arg226_1, (2048, 512), (512, 1))
    assert_size_stride(arg227_1, (2048, ), (1, ))
    assert_size_stride(arg228_1, (512, 2048), (2048, 1))
    assert_size_stride(arg229_1, (512, ), (1, ))
    assert_size_stride(arg230_1, (512, ), (1, ))
    assert_size_stride(arg231_1, (512, ), (1, ))
    assert_size_stride(arg232_1, (1536, 512), (512, 1))
    assert_size_stride(arg233_1, (1536, ), (1, ))
    assert_size_stride(arg234_1, (512, 512), (512, 1))
    assert_size_stride(arg235_1, (512, ), (1, ))
    assert_size_stride(arg236_1, (512, ), (1, ))
    assert_size_stride(arg237_1, (512, ), (1, ))
    assert_size_stride(arg238_1, (2048, 512), (512, 1))
    assert_size_stride(arg239_1, (2048, ), (1, ))
    assert_size_stride(arg240_1, (512, 2048), (2048, 1))
    assert_size_stride(arg241_1, (512, ), (1, ))
    assert_size_stride(arg242_1, (512, ), (1, ))
    assert_size_stride(arg243_1, (512, ), (1, ))
    assert_size_stride(arg244_1, (1536, 512), (512, 1))
    assert_size_stride(arg245_1, (1536, ), (1, ))
    assert_size_stride(arg246_1, (512, 512), (512, 1))
    assert_size_stride(arg247_1, (512, ), (1, ))
    assert_size_stride(arg248_1, (512, ), (1, ))
    assert_size_stride(arg249_1, (512, ), (1, ))
    assert_size_stride(arg250_1, (2048, 512), (512, 1))
    assert_size_stride(arg251_1, (2048, ), (1, ))
    assert_size_stride(arg252_1, (512, 2048), (2048, 1))
    assert_size_stride(arg253_1, (512, ), (1, ))
    assert_size_stride(arg254_1, (512, ), (1, ))
    assert_size_stride(arg255_1, (512, ), (1, ))
    assert_size_stride(arg256_1, (1536, 512), (512, 1))
    assert_size_stride(arg257_1, (1536, ), (1, ))
    assert_size_stride(arg258_1, (512, 512), (512, 1))
    assert_size_stride(arg259_1, (512, ), (1, ))
    assert_size_stride(arg260_1, (512, ), (1, ))
    assert_size_stride(arg261_1, (512, ), (1, ))
    assert_size_stride(arg262_1, (2048, 512), (512, 1))
    assert_size_stride(arg263_1, (2048, ), (1, ))
    assert_size_stride(arg264_1, (512, 2048), (2048, 1))
    assert_size_stride(arg265_1, (512, ), (1, ))
    assert_size_stride(arg266_1, (512, ), (1, ))
    assert_size_stride(arg267_1, (512, ), (1, ))
    assert_size_stride(arg268_1, (1536, 512), (512, 1))
    assert_size_stride(arg269_1, (1536, ), (1, ))
    assert_size_stride(arg270_1, (512, 512), (512, 1))
    assert_size_stride(arg271_1, (512, ), (1, ))
    assert_size_stride(arg272_1, (512, ), (1, ))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (2048, 512), (512, 1))
    assert_size_stride(arg275_1, (2048, ), (1, ))
    assert_size_stride(arg276_1, (512, 2048), (2048, 1))
    assert_size_stride(arg277_1, (512, ), (1, ))
    assert_size_stride(arg278_1, (512, ), (1, ))
    assert_size_stride(arg279_1, (512, ), (1, ))
    assert_size_stride(arg280_1, (1536, 512), (512, 1))
    assert_size_stride(arg281_1, (1536, ), (1, ))
    assert_size_stride(arg282_1, (512, 512), (512, 1))
    assert_size_stride(arg283_1, (512, ), (1, ))
    assert_size_stride(arg284_1, (512, ), (1, ))
    assert_size_stride(arg285_1, (512, ), (1, ))
    assert_size_stride(arg286_1, (2048, 512), (512, 1))
    assert_size_stride(arg287_1, (2048, ), (1, ))
    assert_size_stride(arg288_1, (512, 2048), (2048, 1))
    assert_size_stride(arg289_1, (512, ), (1, ))
    assert_size_stride(arg290_1, (512, ), (1, ))
    assert_size_stride(arg291_1, (512, ), (1, ))
    assert_size_stride(arg292_1, (1536, 512), (512, 1))
    assert_size_stride(arg293_1, (1536, ), (1, ))
    assert_size_stride(arg294_1, (512, 512), (512, 1))
    assert_size_stride(arg295_1, (512, ), (1, ))
    assert_size_stride(arg296_1, (512, ), (1, ))
    assert_size_stride(arg297_1, (512, ), (1, ))
    assert_size_stride(arg298_1, (2048, 512), (512, 1))
    assert_size_stride(arg299_1, (2048, ), (1, ))
    assert_size_stride(arg300_1, (512, 2048), (2048, 1))
    assert_size_stride(arg301_1, (512, ), (1, ))
    assert_size_stride(arg302_1, (512, ), (1, ))
    assert_size_stride(arg303_1, (512, ), (1, ))
    assert_size_stride(arg304_1, (1000, 512), (512, 1))
    assert_size_stride(arg305_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_347], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg0_1
        del arg1_1
        buf4 = empty_strided_cuda((8, 16, 196, 128), (401408, 25088, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_351, x_352], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_0.run(buf0, arg2_1, arg3_1, arg4_1, arg5_1, buf4, 25088, 128, grid=grid(25088), stream=stream0)
        del arg4_1
        del arg5_1
        buf5 = empty_strided_cuda((25088, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (25088, 128), (128, 1), 0), reinterpret_tensor(arg6_1, (128, 384), (1, 128), 0), out=buf5)
        del arg6_1
        buf6 = reinterpret_tensor(buf4, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_353], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_1.run(buf5, arg7_1, buf6, 3211264, grid=grid(3211264), stream=stream0)
        buf7 = empty_strided_cuda((8, 4, 16, 32, 196), (401408, 100352, 6272, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_353], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf5, arg7_1, buf7, 16384, 196, grid=grid(16384, 196), stream=stream0)
        buf8 = empty_strided_cuda((512, 196, 196), (38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_353], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (512, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf7, (512, 32, 196), (6272, 196, 1), 0), out=buf8)
        buf12 = empty_strided_cuda((8, 4, 16, 196, 196), (2458624, 614656, 38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_353], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_3.run(buf8, buf12, 100352, 196, grid=grid(100352), stream=stream0)
        buf13 = reinterpret_tensor(buf7, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_353], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf5, arg7_1, buf13, 3211264, grid=grid(3211264), stream=stream0)
        del arg7_1
        buf14 = reinterpret_tensor(buf6, (512, 196, 32), (6272, 32, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_353], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf12, (512, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf13, (512, 196, 32), (6272, 32, 1), 0), out=buf14)
        buf15 = reinterpret_tensor(buf13, (8, 16, 196, 32, 4), (401408, 25088, 128, 4, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_354], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf14, buf15, 802816, 4, grid=grid(802816, 4), stream=stream0)
        buf16 = reinterpret_tensor(buf14, (25088, 128), (128, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (25088, 128), (128, 1), 0), reinterpret_tensor(arg8_1, (128, 128), (1, 128), 0), out=buf16)
        del arg8_1
        buf17 = reinterpret_tensor(buf16, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf16  # reuse
        buf21 = reinterpret_tensor(buf15, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [x_351, x_357, x_358], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf17, buf0, arg2_1, arg3_1, arg9_1, arg10_1, arg11_1, buf21, 25088, 128, grid=grid(25088), stream=stream0)
        del arg10_1
        del arg11_1
        del arg2_1
        del arg3_1
        del arg9_1
        buf22 = empty_strided_cuda((25088, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (25088, 128), (128, 1), 0), reinterpret_tensor(arg12_1, (128, 512), (1, 128), 0), out=buf22)
        del arg12_1
        buf23 = reinterpret_tensor(buf22, (8, 16, 196, 512), (1605632, 100352, 512, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_360], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf23, arg13_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg13_1
        buf24 = reinterpret_tensor(buf21, (25088, 128), (128, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (25088, 512), (512, 1), 0), reinterpret_tensor(arg14_1, (512, 128), (1, 512), 0), out=buf24)
        del arg14_1
        buf28 = reinterpret_tensor(buf0, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_364, x_365], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf17, buf24, arg15_1, arg16_1, arg17_1, buf28, 25088, 128, grid=grid(25088), stream=stream0)
        del arg16_1
        del arg17_1
        buf29 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (25088, 128), (128, 1), 0), reinterpret_tensor(arg18_1, (128, 384), (1, 128), 0), out=buf29)
        del arg18_1
        buf30 = reinterpret_tensor(buf28, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_366], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_1.run(buf29, arg19_1, buf30, 3211264, grid=grid(3211264), stream=stream0)
        buf31 = empty_strided_cuda((8, 4, 16, 32, 196), (401408, 100352, 6272, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_366], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_2.run(buf29, arg19_1, buf31, 16384, 196, grid=grid(16384, 196), stream=stream0)
        buf32 = reinterpret_tensor(buf12, (512, 196, 196), (38416, 196, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_366], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf30, (512, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf31, (512, 32, 196), (6272, 196, 1), 0), out=buf32)
        buf36 = reinterpret_tensor(buf8, (8, 4, 16, 196, 196), (2458624, 614656, 38416, 196, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_366], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_3.run(buf32, buf36, 100352, 196, grid=grid(100352), stream=stream0)
        del buf32
        buf37 = reinterpret_tensor(buf31, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_366], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf29, arg19_1, buf37, 3211264, grid=grid(3211264), stream=stream0)
        del arg19_1
        del buf29
        buf38 = reinterpret_tensor(buf30, (512, 196, 32), (6272, 32, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_366], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (512, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf37, (512, 196, 32), (6272, 32, 1), 0), out=buf38)
        del buf36
        buf39 = reinterpret_tensor(buf37, (8, 16, 196, 32, 4), (401408, 25088, 128, 4, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_367], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf38, buf39, 802816, 4, grid=grid(802816, 4), stream=stream0)
        buf40 = reinterpret_tensor(buf38, (25088, 128), (128, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (25088, 128), (128, 1), 0), reinterpret_tensor(arg20_1, (128, 128), (1, 128), 0), out=buf40)
        del arg20_1
        buf41 = reinterpret_tensor(buf40, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf40  # reuse
        buf45 = reinterpret_tensor(buf39, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_364, x_370, x_371], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf41, buf17, buf24, arg15_1, arg21_1, arg22_1, arg23_1, buf45, 25088, 128, grid=grid(25088), stream=stream0)
        del arg15_1
        del arg21_1
        del arg22_1
        del arg23_1
        del buf17
        buf46 = reinterpret_tensor(buf23, (25088, 512), (512, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf45, (25088, 128), (128, 1), 0), reinterpret_tensor(arg24_1, (128, 512), (1, 128), 0), out=buf46)
        del arg24_1
        buf47 = reinterpret_tensor(buf46, (8, 16, 196, 512), (1605632, 100352, 512, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_373], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf47, arg25_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg25_1
        buf48 = reinterpret_tensor(buf45, (25088, 128), (128, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (25088, 512), (512, 1), 0), reinterpret_tensor(arg26_1, (512, 128), (1, 512), 0), out=buf48)
        del arg26_1
        del buf47
        buf49 = reinterpret_tensor(buf24, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_380], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf41, buf48, arg27_1, buf49, 3211264, grid=grid(3211264), stream=stream0)
        del arg27_1
        del buf41
        del buf48
        buf50 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_380], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(arg28_1, buf50, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg28_1
        # Topologically Sorted Source Nodes: [x_380], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf49, buf50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del buf49
        del buf50
        buf52 = empty_strided_cuda((8, 56, 56, 1), (3136, 56, 1, 25088), torch.float32)
        buf53 = empty_strided_cuda((8, 56, 56, 1), (3136, 56, 1, 25088), torch.float32)
        # Topologically Sorted Source Nodes: [x_381], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_12.run(buf51, arg29_1, buf52, buf53, 25088, 256, grid=grid(25088), stream=stream0)
        buf55 = empty_strided_cuda((8, 256, 57, 57), (831744, 1, 14592, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_383], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_13.run(buf51, arg29_1, buf52, buf53, arg30_1, arg31_1, buf55, 6653952, grid=grid(6653952), stream=stream0)
        del arg29_1
        del arg30_1
        del arg31_1
        del buf52
        del buf53
        buf56 = empty_strided_cuda((8, 256, 28, 28), (200704, 784, 28, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_383, x_384], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_14.run(buf55, buf56, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del buf55
        buf57 = empty_strided_cuda((8, 4, 196, 1, 2), (1568, 392, 2, 12544, 1), torch.float32)
        buf58 = empty_strided_cuda((8, 4, 196, 1, 2), (1568, 392, 2, 12544, 1), torch.float32)
        buf59 = empty_strided_cuda((8, 4, 196, 1, 2), (1568, 392, 2, 12544, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_388, x_389], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_15.run(buf56, arg32_1, buf57, buf58, buf59, 12544, 128, grid=grid(12544), stream=stream0)
        buf60 = empty_strided_cuda((8, 4, 196, 1), (784, 196, 1, 6272), torch.float32)
        buf61 = empty_strided_cuda((8, 4, 196, 1), (784, 196, 1, 6272), torch.float32)
        # Topologically Sorted Source Nodes: [x_388, x_389], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf57, buf58, buf59, buf60, buf61, 6272, 2, grid=grid(6272), stream=stream0)
        buf63 = empty_strided_cuda((8, 4, 196, 256), (200704, 50176, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_388, x_389], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_17.run(buf56, arg32_1, buf60, buf61, arg33_1, arg34_1, buf63, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg33_1
        del arg34_1
        buf64 = empty_strided_cuda((6272, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (6272, 256), (256, 1), 0), reinterpret_tensor(arg35_1, (256, 768), (1, 256), 0), out=buf64)
        del arg35_1
        buf65 = reinterpret_tensor(buf63, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [x_390], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_18.run(buf64, arg36_1, buf65, 1605632, grid=grid(1605632), stream=stream0)
        buf66 = empty_strided_cuda((8, 8, 4, 32, 196), (200704, 25088, 6272, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_390], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_19.run(buf64, arg36_1, buf66, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf67 = empty_strided_cuda((256, 196, 196), (38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_390], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf65, (256, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf66, (256, 32, 196), (6272, 196, 1), 0), out=buf67)
        buf71 = empty_strided_cuda((8, 8, 4, 196, 196), (1229312, 153664, 38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_390], Original ATen: [aten._safe_softmax]
        triton_per_fused__safe_softmax_20.run(buf67, buf71, 50176, 196, grid=grid(50176), stream=stream0)
        buf72 = reinterpret_tensor(buf66, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_390], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf64, arg36_1, buf72, 1605632, grid=grid(1605632), stream=stream0)
        del arg36_1
        buf73 = reinterpret_tensor(buf65, (256, 196, 32), (6272, 32, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_390], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf71, (256, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf72, (256, 196, 32), (6272, 32, 1), 0), out=buf73)
        buf74 = reinterpret_tensor(buf72, (8, 4, 196, 32, 8), (200704, 50176, 256, 8, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_391], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf73, buf74, 200704, 8, grid=grid(200704, 8), stream=stream0)
        buf75 = reinterpret_tensor(buf73, (6272, 256), (256, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (6272, 256), (256, 1), 0), reinterpret_tensor(arg37_1, (256, 256), (1, 256), 0), out=buf75)
        del arg37_1
        buf76 = buf59; del buf59  # reuse
        buf77 = buf58; del buf58  # reuse
        buf78 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_388, x_394, x_395], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_23.run(buf56, arg32_1, buf75, arg38_1, buf76, buf77, buf78, 12544, 128, grid=grid(12544), stream=stream0)
        buf79 = buf61; del buf61  # reuse
        buf80 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_388, x_394, x_395], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf76, buf77, buf78, buf79, buf80, 6272, 2, grid=grid(6272), stream=stream0)
        del buf76
        del buf77
        del buf78
        buf82 = reinterpret_tensor(buf74, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_388, x_394, x_395], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_24.run(buf56, arg32_1, buf75, arg38_1, buf79, buf80, arg39_1, arg40_1, buf82, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg39_1
        del arg40_1
        buf83 = reinterpret_tensor(buf51, (6272, 1024), (1024, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (6272, 256), (256, 1), 0), reinterpret_tensor(arg41_1, (256, 1024), (1, 256), 0), out=buf83)
        del arg41_1
        buf84 = reinterpret_tensor(buf83, (8, 4, 196, 1024), (802816, 200704, 1024, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_397], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf84, arg42_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg42_1
        buf85 = reinterpret_tensor(buf82, (6272, 256), (256, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg43_1, (1024, 256), (1, 1024), 0), out=buf85)
        del arg43_1
        buf86 = reinterpret_tensor(buf75, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf75  # reuse
        buf90 = empty_strided_cuda((8, 4, 196, 256), (200704, 50176, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_388, x_394, x_401, x_402], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_26.run(buf86, buf56, arg32_1, arg38_1, buf85, arg44_1, arg45_1, arg46_1, buf90, 6272, 256, grid=grid(6272), stream=stream0)
        del arg32_1
        del arg38_1
        del arg44_1
        del arg45_1
        del arg46_1
        buf91 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (6272, 256), (256, 1), 0), reinterpret_tensor(arg47_1, (256, 768), (1, 256), 0), out=buf91)
        del arg47_1
        buf92 = reinterpret_tensor(buf90, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_403], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_18.run(buf91, arg48_1, buf92, 1605632, grid=grid(1605632), stream=stream0)
        buf93 = reinterpret_tensor(buf85, (8, 8, 4, 32, 196), (200704, 25088, 6272, 196, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_403], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_19.run(buf91, arg48_1, buf93, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf94 = reinterpret_tensor(buf71, (256, 196, 196), (38416, 196, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_403], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf92, (256, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf93, (256, 32, 196), (6272, 196, 1), 0), out=buf94)
        buf98 = reinterpret_tensor(buf67, (8, 8, 4, 196, 196), (1229312, 153664, 38416, 196, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_403], Original ATen: [aten._safe_softmax]
        triton_per_fused__safe_softmax_20.run(buf94, buf98, 50176, 196, grid=grid(50176), stream=stream0)
        del buf94
        buf99 = reinterpret_tensor(buf93, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_403], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf91, arg48_1, buf99, 1605632, grid=grid(1605632), stream=stream0)
        del arg48_1
        del buf91
        buf100 = reinterpret_tensor(buf92, (256, 196, 32), (6272, 32, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_403], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf98, (256, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf99, (256, 196, 32), (6272, 32, 1), 0), out=buf100)
        del buf98
        buf101 = reinterpret_tensor(buf99, (8, 4, 196, 32, 8), (200704, 50176, 256, 8, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [x_404], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf100, buf101, 200704, 8, grid=grid(200704, 8), stream=stream0)
        buf102 = reinterpret_tensor(buf100, (6272, 256), (256, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (6272, 256), (256, 1), 0), reinterpret_tensor(arg49_1, (256, 256), (1, 256), 0), out=buf102)
        del arg49_1
        buf106 = reinterpret_tensor(buf101, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_407, x_408], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf86, buf102, arg50_1, arg51_1, arg52_1, buf106, 6272, 256, grid=grid(6272), stream=stream0)
        del arg51_1
        del arg52_1
        buf107 = reinterpret_tensor(buf84, (6272, 1024), (1024, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf106, (6272, 256), (256, 1), 0), reinterpret_tensor(arg53_1, (256, 1024), (1, 256), 0), out=buf107)
        del arg53_1
        buf108 = reinterpret_tensor(buf107, (8, 4, 196, 1024), (802816, 200704, 1024, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [x_410], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf108, arg54_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg54_1
        buf109 = reinterpret_tensor(buf106, (6272, 256), (256, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf108, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg55_1, (1024, 256), (1, 1024), 0), out=buf109)
        del arg55_1
        del buf108
        buf110 = reinterpret_tensor(buf56, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_417], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf86, buf102, arg50_1, buf109, arg56_1, buf110, 1605632, grid=grid(1605632), stream=stream0)
        del arg50_1
        del arg56_1
        del buf102
        del buf109
        del buf86
        buf111 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_417], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(arg57_1, buf111, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg57_1
        # Topologically Sorted Source Nodes: [x_417], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf110, buf111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del buf110
        del buf111
        buf113 = reinterpret_tensor(buf80, (8, 28, 28, 1), (784, 28, 1, 6272), 0); del buf80  # reuse
        buf114 = reinterpret_tensor(buf79, (8, 28, 28, 1), (784, 28, 1, 6272), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [x_418], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_30.run(buf112, arg58_1, buf113, buf114, 6272, 512, grid=grid(6272), stream=stream0)
        buf116 = empty_strided_cuda((8, 512, 29, 29), (430592, 1, 14848, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_420], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_31.run(buf112, arg58_1, buf113, buf114, arg59_1, arg60_1, buf116, 3444736, grid=grid(3444736), stream=stream0)
        del arg58_1
        del arg59_1
        del arg60_1
        buf117 = empty_strided_cuda((8, 512, 14, 14), (100352, 196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_420, x_421], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_32.run(buf116, buf117, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del buf116
        buf118 = reinterpret_tensor(buf114, (8, 1, 196, 1, 4), (784, 6272, 4, 6272, 1), 0); del buf114  # reuse
        buf119 = reinterpret_tensor(buf113, (8, 1, 196, 1, 4), (784, 6272, 4, 6272, 1), 0); del buf113  # reuse
        buf120 = empty_strided_cuda((8, 1, 196, 1, 4), (784, 6272, 4, 6272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_425, x_426], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_33.run(buf117, arg61_1, buf118, buf119, buf120, 6272, 128, grid=grid(6272), stream=stream0)
        buf121 = empty_strided_cuda((8, 1, 196, 1), (196, 1568, 1, 1568), torch.float32)
        buf122 = empty_strided_cuda((8, 1, 196, 1), (196, 1568, 1, 1568), torch.float32)
        # Topologically Sorted Source Nodes: [x_425, x_426], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_34.run(buf118, buf119, buf120, buf121, buf122, 1568, 4, grid=grid(1568), stream=stream0)
        buf124 = empty_strided_cuda((8, 1, 196, 512), (100352, 1, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_425, x_426], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_35.run(buf117, arg61_1, buf121, buf122, arg62_1, arg63_1, buf124, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg62_1
        del arg63_1
        buf125 = empty_strided_cuda((1568, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (1568, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 1536), (1, 512), 0), out=buf125)
        del arg64_1
        buf126 = reinterpret_tensor(buf124, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_427], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf125, arg65_1, buf126, 802816, grid=grid(802816), stream=stream0)
        buf127 = empty_strided_cuda((8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_427], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf125, arg65_1, buf127, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf128 = empty_strided_cuda((128, 196, 196), (38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_427], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf126, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf127, (128, 32, 196), (6272, 196, 1), 0), out=buf128)
        buf132 = empty_strided_cuda((8, 16, 1, 196, 196), (614912, 38432, 38432, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_427], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf128, buf132, 25088, 196, grid=grid(25088), stream=stream0)
        buf133 = reinterpret_tensor(buf127, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_427], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf125, arg65_1, buf133, 802816, grid=grid(802816), stream=stream0)
        del arg65_1
        buf134 = reinterpret_tensor(buf126, (128, 196, 32), (6272, 32, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_427], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf133, (128, 196, 32), (6272, 32, 1), 0), out=buf134)
        buf135 = reinterpret_tensor(buf133, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [x_428], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf134, buf135, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf136 = reinterpret_tensor(buf134, (1568, 512), (512, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (1568, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 512), (1, 512), 0), out=buf136)
        del arg66_1
        buf137 = buf120; del buf120  # reuse
        buf138 = buf119; del buf119  # reuse
        buf139 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_425, x_431, x_432], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_41.run(buf117, arg61_1, buf136, arg67_1, buf137, buf138, buf139, 6272, 128, grid=grid(6272), stream=stream0)
        buf140 = buf122; del buf122  # reuse
        buf141 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [x_425, x_431, x_432], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_34.run(buf137, buf138, buf139, buf140, buf141, 1568, 4, grid=grid(1568), stream=stream0)
        del buf137
        del buf138
        del buf139
        buf143 = reinterpret_tensor(buf135, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [x_425, x_431, x_432], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_42.run(buf117, arg61_1, buf136, arg67_1, buf140, buf141, arg68_1, arg69_1, buf143, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg68_1
        del arg69_1
        buf144 = reinterpret_tensor(buf112, (1568, 2048), (2048, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (1568, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 2048), (1, 512), 0), out=buf144)
        del arg70_1
        buf145 = reinterpret_tensor(buf144, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_434], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf145, arg71_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg71_1
        buf146 = reinterpret_tensor(buf143, (1568, 512), (512, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg72_1, (2048, 512), (1, 2048), 0), out=buf146)
        del arg72_1
        buf147 = reinterpret_tensor(buf136, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf136  # reuse
        buf151 = empty_strided_cuda((8, 1, 196, 512), (100352, 1, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_425, x_431, x_438, x_439], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_44.run(buf147, buf117, arg61_1, arg67_1, buf146, arg73_1, arg74_1, arg75_1, buf151, 1568, 512, grid=grid(1568), stream=stream0)
        del arg61_1
        del arg67_1
        del arg73_1
        del arg74_1
        del arg75_1
        buf152 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (1568, 512), (512, 1), 0), reinterpret_tensor(arg76_1, (512, 1536), (1, 512), 0), out=buf152)
        del arg76_1
        buf153 = reinterpret_tensor(buf151, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [x_440], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf152, arg77_1, buf153, 802816, grid=grid(802816), stream=stream0)
        buf154 = reinterpret_tensor(buf146, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_440], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf152, arg77_1, buf154, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf155 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_440], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf153, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf154, (128, 32, 196), (6272, 196, 1), 0), out=buf155)
        buf159 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [x_440], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf155, buf159, 25088, 196, grid=grid(25088), stream=stream0)
        buf160 = reinterpret_tensor(buf154, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [x_440], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf152, arg77_1, buf160, 802816, grid=grid(802816), stream=stream0)
        del arg77_1
        buf161 = reinterpret_tensor(buf153, (128, 196, 32), (6272, 32, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [x_440], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf159, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf160, (128, 196, 32), (6272, 32, 1), 0), out=buf161)
        buf162 = reinterpret_tensor(buf160, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_441], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf161, buf162, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf163 = reinterpret_tensor(buf161, (1568, 512), (512, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf162, (1568, 512), (512, 1), 0), reinterpret_tensor(arg78_1, (512, 512), (1, 512), 0), out=buf163)
        del arg78_1
        buf167 = reinterpret_tensor(buf162, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_444, x_445], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf147, buf163, arg79_1, arg80_1, arg81_1, buf167, 1568, 512, grid=grid(1568), stream=stream0)
        del arg80_1
        del arg81_1
        buf168 = reinterpret_tensor(buf145, (1568, 2048), (2048, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (1568, 512), (512, 1), 0), reinterpret_tensor(arg82_1, (512, 2048), (1, 512), 0), out=buf168)
        del arg82_1
        buf169 = reinterpret_tensor(buf168, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_447], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf169, arg83_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg83_1
        buf170 = reinterpret_tensor(buf167, (1568, 512), (512, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg84_1, (2048, 512), (1, 2048), 0), out=buf170)
        del arg84_1
        buf171 = reinterpret_tensor(buf170, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf170  # reuse
        buf175 = reinterpret_tensor(buf117, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [x_444, x_451, x_452], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf171, buf147, buf163, arg79_1, arg85_1, arg86_1, arg87_1, buf175, 1568, 512, grid=grid(1568), stream=stream0)
        del arg79_1
        del arg85_1
        del arg86_1
        del arg87_1
        buf176 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf175, (1568, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 1536), (1, 512), 0), out=buf176)
        del arg88_1
        buf177 = reinterpret_tensor(buf175, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [x_453], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf176, arg89_1, buf177, 802816, grid=grid(802816), stream=stream0)
        buf178 = reinterpret_tensor(buf163, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_453], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf176, arg89_1, buf178, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf179 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_453], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf177, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf178, (128, 32, 196), (6272, 196, 1), 0), out=buf179)
        buf183 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_453], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf179, buf183, 25088, 196, grid=grid(25088), stream=stream0)
        buf184 = reinterpret_tensor(buf178, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_453], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf176, arg89_1, buf184, 802816, grid=grid(802816), stream=stream0)
        del arg89_1
        buf185 = reinterpret_tensor(buf177, (128, 196, 32), (6272, 32, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_453], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf183, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf184, (128, 196, 32), (6272, 32, 1), 0), out=buf185)
        buf186 = reinterpret_tensor(buf184, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [x_454], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf185, buf186, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf187 = reinterpret_tensor(buf185, (1568, 512), (512, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf186, (1568, 512), (512, 1), 0), reinterpret_tensor(arg90_1, (512, 512), (1, 512), 0), out=buf187)
        del arg90_1
        buf191 = reinterpret_tensor(buf186, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_457, x_458], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf171, buf187, arg91_1, arg92_1, arg93_1, buf191, 1568, 512, grid=grid(1568), stream=stream0)
        del arg92_1
        del arg93_1
        buf192 = reinterpret_tensor(buf169, (1568, 2048), (2048, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (1568, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 2048), (1, 512), 0), out=buf192)
        del arg94_1
        buf193 = reinterpret_tensor(buf192, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [x_460], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf193, arg95_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg95_1
        buf194 = reinterpret_tensor(buf191, (1568, 512), (512, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf193, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg96_1, (2048, 512), (1, 2048), 0), out=buf194)
        del arg96_1
        buf195 = reinterpret_tensor(buf194, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf194  # reuse
        buf199 = reinterpret_tensor(buf147, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_457, x_464, x_465], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf195, buf171, buf187, arg91_1, arg97_1, arg98_1, arg99_1, buf199, 1568, 512, grid=grid(1568), stream=stream0)
        del arg91_1
        del arg97_1
        del arg98_1
        del arg99_1
        buf200 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (1568, 512), (512, 1), 0), reinterpret_tensor(arg100_1, (512, 1536), (1, 512), 0), out=buf200)
        del arg100_1
        buf201 = reinterpret_tensor(buf199, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [x_466], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf200, arg101_1, buf201, 802816, grid=grid(802816), stream=stream0)
        buf202 = reinterpret_tensor(buf187, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [x_466], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf200, arg101_1, buf202, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf203 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_466], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf201, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf202, (128, 32, 196), (6272, 196, 1), 0), out=buf203)
        buf207 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [x_466], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf203, buf207, 25088, 196, grid=grid(25088), stream=stream0)
        buf208 = reinterpret_tensor(buf202, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [x_466], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf200, arg101_1, buf208, 802816, grid=grid(802816), stream=stream0)
        del arg101_1
        buf209 = reinterpret_tensor(buf201, (128, 196, 32), (6272, 32, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [x_466], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf207, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf208, (128, 196, 32), (6272, 32, 1), 0), out=buf209)
        buf210 = reinterpret_tensor(buf208, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [x_467], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf209, buf210, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf211 = reinterpret_tensor(buf209, (1568, 512), (512, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (1568, 512), (512, 1), 0), reinterpret_tensor(arg102_1, (512, 512), (1, 512), 0), out=buf211)
        del arg102_1
        buf215 = reinterpret_tensor(buf210, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [x_470, x_471], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf195, buf211, arg103_1, arg104_1, arg105_1, buf215, 1568, 512, grid=grid(1568), stream=stream0)
        del arg104_1
        del arg105_1
        buf216 = reinterpret_tensor(buf193, (1568, 2048), (2048, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (1568, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 2048), (1, 512), 0), out=buf216)
        del arg106_1
        buf217 = reinterpret_tensor(buf216, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [x_473], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf217, arg107_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg107_1
        buf218 = reinterpret_tensor(buf215, (1568, 512), (512, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg108_1, (2048, 512), (1, 2048), 0), out=buf218)
        del arg108_1
        buf219 = reinterpret_tensor(buf218, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf218  # reuse
        buf223 = reinterpret_tensor(buf171, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_470, x_477, x_478], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf219, buf195, buf211, arg103_1, arg109_1, arg110_1, arg111_1, buf223, 1568, 512, grid=grid(1568), stream=stream0)
        del arg103_1
        del arg109_1
        del arg110_1
        del arg111_1
        buf224 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (1568, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 1536), (1, 512), 0), out=buf224)
        del arg112_1
        buf225 = reinterpret_tensor(buf223, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [x_479], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf224, arg113_1, buf225, 802816, grid=grid(802816), stream=stream0)
        buf226 = reinterpret_tensor(buf211, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [x_479], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf224, arg113_1, buf226, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf227 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [x_479], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf225, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf226, (128, 32, 196), (6272, 196, 1), 0), out=buf227)
        buf231 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_479], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf227, buf231, 25088, 196, grid=grid(25088), stream=stream0)
        buf232 = reinterpret_tensor(buf226, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [x_479], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf224, arg113_1, buf232, 802816, grid=grid(802816), stream=stream0)
        del arg113_1
        buf233 = reinterpret_tensor(buf225, (128, 196, 32), (6272, 32, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [x_479], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf231, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf232, (128, 196, 32), (6272, 32, 1), 0), out=buf233)
        buf234 = reinterpret_tensor(buf232, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_480], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf233, buf234, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf235 = reinterpret_tensor(buf233, (1568, 512), (512, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf234, (1568, 512), (512, 1), 0), reinterpret_tensor(arg114_1, (512, 512), (1, 512), 0), out=buf235)
        del arg114_1
        buf239 = reinterpret_tensor(buf234, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [x_483, x_484], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf219, buf235, arg115_1, arg116_1, arg117_1, buf239, 1568, 512, grid=grid(1568), stream=stream0)
        del arg116_1
        del arg117_1
        buf240 = reinterpret_tensor(buf217, (1568, 2048), (2048, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (1568, 512), (512, 1), 0), reinterpret_tensor(arg118_1, (512, 2048), (1, 512), 0), out=buf240)
        del arg118_1
        buf241 = reinterpret_tensor(buf240, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [x_486], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf241, arg119_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg119_1
        buf242 = reinterpret_tensor(buf239, (1568, 512), (512, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf241, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg120_1, (2048, 512), (1, 2048), 0), out=buf242)
        del arg120_1
        buf243 = reinterpret_tensor(buf242, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf242  # reuse
        buf247 = reinterpret_tensor(buf195, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_483, x_490, x_491], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf243, buf219, buf235, arg115_1, arg121_1, arg122_1, arg123_1, buf247, 1568, 512, grid=grid(1568), stream=stream0)
        del arg115_1
        del arg121_1
        del arg122_1
        del arg123_1
        buf248 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (1568, 512), (512, 1), 0), reinterpret_tensor(arg124_1, (512, 1536), (1, 512), 0), out=buf248)
        del arg124_1
        buf249 = reinterpret_tensor(buf247, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [x_492], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf248, arg125_1, buf249, 802816, grid=grid(802816), stream=stream0)
        buf250 = reinterpret_tensor(buf235, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [x_492], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf248, arg125_1, buf250, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf251 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [x_492], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf249, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf250, (128, 32, 196), (6272, 196, 1), 0), out=buf251)
        buf255 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [x_492], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf251, buf255, 25088, 196, grid=grid(25088), stream=stream0)
        buf256 = reinterpret_tensor(buf250, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [x_492], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf248, arg125_1, buf256, 802816, grid=grid(802816), stream=stream0)
        del arg125_1
        buf257 = reinterpret_tensor(buf249, (128, 196, 32), (6272, 32, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [x_492], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf255, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf256, (128, 196, 32), (6272, 32, 1), 0), out=buf257)
        buf258 = reinterpret_tensor(buf256, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [x_493], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf257, buf258, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf259 = reinterpret_tensor(buf257, (1568, 512), (512, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (1568, 512), (512, 1), 0), reinterpret_tensor(arg126_1, (512, 512), (1, 512), 0), out=buf259)
        del arg126_1
        buf263 = reinterpret_tensor(buf258, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [x_496, x_497], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf243, buf259, arg127_1, arg128_1, arg129_1, buf263, 1568, 512, grid=grid(1568), stream=stream0)
        del arg128_1
        del arg129_1
        buf264 = reinterpret_tensor(buf241, (1568, 2048), (2048, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf263, (1568, 512), (512, 1), 0), reinterpret_tensor(arg130_1, (512, 2048), (1, 512), 0), out=buf264)
        del arg130_1
        buf265 = reinterpret_tensor(buf264, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [x_499], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf265, arg131_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg131_1
        buf266 = reinterpret_tensor(buf263, (1568, 512), (512, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf265, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg132_1, (2048, 512), (1, 2048), 0), out=buf266)
        del arg132_1
        buf267 = reinterpret_tensor(buf266, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf266  # reuse
        buf271 = reinterpret_tensor(buf219, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_496, x_503, x_504], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf267, buf243, buf259, arg127_1, arg133_1, arg134_1, arg135_1, buf271, 1568, 512, grid=grid(1568), stream=stream0)
        del arg127_1
        del arg133_1
        del arg134_1
        del arg135_1
        buf272 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (1568, 512), (512, 1), 0), reinterpret_tensor(arg136_1, (512, 1536), (1, 512), 0), out=buf272)
        del arg136_1
        buf273 = reinterpret_tensor(buf271, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [x_505], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf272, arg137_1, buf273, 802816, grid=grid(802816), stream=stream0)
        buf274 = reinterpret_tensor(buf259, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [x_505], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf272, arg137_1, buf274, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf275 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [x_505], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf273, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf274, (128, 32, 196), (6272, 196, 1), 0), out=buf275)
        buf279 = buf255; del buf255  # reuse
        # Topologically Sorted Source Nodes: [x_505], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf275, buf279, 25088, 196, grid=grid(25088), stream=stream0)
        buf280 = reinterpret_tensor(buf274, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [x_505], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf272, arg137_1, buf280, 802816, grid=grid(802816), stream=stream0)
        del arg137_1
        buf281 = reinterpret_tensor(buf273, (128, 196, 32), (6272, 32, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [x_505], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf279, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf280, (128, 196, 32), (6272, 32, 1), 0), out=buf281)
        buf282 = reinterpret_tensor(buf280, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [x_506], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf281, buf282, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf283 = reinterpret_tensor(buf281, (1568, 512), (512, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (1568, 512), (512, 1), 0), reinterpret_tensor(arg138_1, (512, 512), (1, 512), 0), out=buf283)
        del arg138_1
        buf287 = reinterpret_tensor(buf282, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [x_509, x_510], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf267, buf283, arg139_1, arg140_1, arg141_1, buf287, 1568, 512, grid=grid(1568), stream=stream0)
        del arg140_1
        del arg141_1
        buf288 = reinterpret_tensor(buf265, (1568, 2048), (2048, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf287, (1568, 512), (512, 1), 0), reinterpret_tensor(arg142_1, (512, 2048), (1, 512), 0), out=buf288)
        del arg142_1
        buf289 = reinterpret_tensor(buf288, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [x_512], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf289, arg143_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg143_1
        buf290 = reinterpret_tensor(buf287, (1568, 512), (512, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf289, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg144_1, (2048, 512), (1, 2048), 0), out=buf290)
        del arg144_1
        buf291 = reinterpret_tensor(buf290, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf290  # reuse
        buf295 = reinterpret_tensor(buf243, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [x_509, x_516, x_517], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf291, buf267, buf283, arg139_1, arg145_1, arg146_1, arg147_1, buf295, 1568, 512, grid=grid(1568), stream=stream0)
        del arg139_1
        del arg145_1
        del arg146_1
        del arg147_1
        buf296 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf295, (1568, 512), (512, 1), 0), reinterpret_tensor(arg148_1, (512, 1536), (1, 512), 0), out=buf296)
        del arg148_1
        buf297 = reinterpret_tensor(buf295, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [x_518], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf296, arg149_1, buf297, 802816, grid=grid(802816), stream=stream0)
        buf298 = reinterpret_tensor(buf283, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [x_518], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf296, arg149_1, buf298, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf299 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [x_518], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf297, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf298, (128, 32, 196), (6272, 196, 1), 0), out=buf299)
        buf303 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [x_518], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf299, buf303, 25088, 196, grid=grid(25088), stream=stream0)
        buf304 = reinterpret_tensor(buf298, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf298  # reuse
        # Topologically Sorted Source Nodes: [x_518], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf296, arg149_1, buf304, 802816, grid=grid(802816), stream=stream0)
        del arg149_1
        buf305 = reinterpret_tensor(buf297, (128, 196, 32), (6272, 32, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [x_518], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf303, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf304, (128, 196, 32), (6272, 32, 1), 0), out=buf305)
        buf306 = reinterpret_tensor(buf304, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [x_519], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf305, buf306, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf307 = reinterpret_tensor(buf305, (1568, 512), (512, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf306, (1568, 512), (512, 1), 0), reinterpret_tensor(arg150_1, (512, 512), (1, 512), 0), out=buf307)
        del arg150_1
        buf311 = reinterpret_tensor(buf306, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [x_522, x_523], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf291, buf307, arg151_1, arg152_1, arg153_1, buf311, 1568, 512, grid=grid(1568), stream=stream0)
        del arg152_1
        del arg153_1
        buf312 = reinterpret_tensor(buf289, (1568, 2048), (2048, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf311, (1568, 512), (512, 1), 0), reinterpret_tensor(arg154_1, (512, 2048), (1, 512), 0), out=buf312)
        del arg154_1
        buf313 = reinterpret_tensor(buf312, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [x_525], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf313, arg155_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg155_1
        buf314 = reinterpret_tensor(buf311, (1568, 512), (512, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg156_1, (2048, 512), (1, 2048), 0), out=buf314)
        del arg156_1
        buf315 = reinterpret_tensor(buf314, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf314  # reuse
        buf319 = reinterpret_tensor(buf267, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [x_522, x_529, x_530], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf315, buf291, buf307, arg151_1, arg157_1, arg158_1, arg159_1, buf319, 1568, 512, grid=grid(1568), stream=stream0)
        del arg151_1
        del arg157_1
        del arg158_1
        del arg159_1
        buf320 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf319, (1568, 512), (512, 1), 0), reinterpret_tensor(arg160_1, (512, 1536), (1, 512), 0), out=buf320)
        del arg160_1
        buf321 = reinterpret_tensor(buf319, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [x_531], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf320, arg161_1, buf321, 802816, grid=grid(802816), stream=stream0)
        buf322 = reinterpret_tensor(buf307, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [x_531], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf320, arg161_1, buf322, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf323 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [x_531], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf321, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf322, (128, 32, 196), (6272, 196, 1), 0), out=buf323)
        buf327 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [x_531], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf323, buf327, 25088, 196, grid=grid(25088), stream=stream0)
        buf328 = reinterpret_tensor(buf322, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [x_531], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf320, arg161_1, buf328, 802816, grid=grid(802816), stream=stream0)
        del arg161_1
        buf329 = reinterpret_tensor(buf321, (128, 196, 32), (6272, 32, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [x_531], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf327, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf328, (128, 196, 32), (6272, 32, 1), 0), out=buf329)
        buf330 = reinterpret_tensor(buf328, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [x_532], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf329, buf330, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf331 = reinterpret_tensor(buf329, (1568, 512), (512, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf330, (1568, 512), (512, 1), 0), reinterpret_tensor(arg162_1, (512, 512), (1, 512), 0), out=buf331)
        del arg162_1
        buf335 = reinterpret_tensor(buf330, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [x_535, x_536], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf315, buf331, arg163_1, arg164_1, arg165_1, buf335, 1568, 512, grid=grid(1568), stream=stream0)
        del arg164_1
        del arg165_1
        buf336 = reinterpret_tensor(buf313, (1568, 2048), (2048, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1568, 512), (512, 1), 0), reinterpret_tensor(arg166_1, (512, 2048), (1, 512), 0), out=buf336)
        del arg166_1
        buf337 = reinterpret_tensor(buf336, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [x_538], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf337, arg167_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg167_1
        buf338 = reinterpret_tensor(buf335, (1568, 512), (512, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf337, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg168_1, (2048, 512), (1, 2048), 0), out=buf338)
        del arg168_1
        buf339 = reinterpret_tensor(buf338, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf338  # reuse
        buf343 = reinterpret_tensor(buf291, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [x_535, x_542, x_543], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf339, buf315, buf331, arg163_1, arg169_1, arg170_1, arg171_1, buf343, 1568, 512, grid=grid(1568), stream=stream0)
        del arg163_1
        del arg169_1
        del arg170_1
        del arg171_1
        buf344 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf343, (1568, 512), (512, 1), 0), reinterpret_tensor(arg172_1, (512, 1536), (1, 512), 0), out=buf344)
        del arg172_1
        buf345 = reinterpret_tensor(buf343, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [x_544], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf344, arg173_1, buf345, 802816, grid=grid(802816), stream=stream0)
        buf346 = reinterpret_tensor(buf331, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [x_544], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf344, arg173_1, buf346, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf347 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [x_544], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf345, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf346, (128, 32, 196), (6272, 196, 1), 0), out=buf347)
        buf351 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [x_544], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf347, buf351, 25088, 196, grid=grid(25088), stream=stream0)
        buf352 = reinterpret_tensor(buf346, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [x_544], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf344, arg173_1, buf352, 802816, grid=grid(802816), stream=stream0)
        del arg173_1
        buf353 = reinterpret_tensor(buf345, (128, 196, 32), (6272, 32, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [x_544], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf351, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf352, (128, 196, 32), (6272, 32, 1), 0), out=buf353)
        buf354 = reinterpret_tensor(buf352, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [x_545], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf353, buf354, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf355 = reinterpret_tensor(buf353, (1568, 512), (512, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (1568, 512), (512, 1), 0), reinterpret_tensor(arg174_1, (512, 512), (1, 512), 0), out=buf355)
        del arg174_1
        buf359 = reinterpret_tensor(buf354, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [x_548, x_549], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf339, buf355, arg175_1, arg176_1, arg177_1, buf359, 1568, 512, grid=grid(1568), stream=stream0)
        del arg176_1
        del arg177_1
        buf360 = reinterpret_tensor(buf337, (1568, 2048), (2048, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf359, (1568, 512), (512, 1), 0), reinterpret_tensor(arg178_1, (512, 2048), (1, 512), 0), out=buf360)
        del arg178_1
        buf361 = reinterpret_tensor(buf360, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [x_551], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf361, arg179_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg179_1
        buf362 = reinterpret_tensor(buf359, (1568, 512), (512, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf361, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg180_1, (2048, 512), (1, 2048), 0), out=buf362)
        del arg180_1
        buf363 = reinterpret_tensor(buf362, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf362  # reuse
        buf367 = reinterpret_tensor(buf315, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [x_548, x_555, x_556], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf363, buf339, buf355, arg175_1, arg181_1, arg182_1, arg183_1, buf367, 1568, 512, grid=grid(1568), stream=stream0)
        del arg175_1
        del arg181_1
        del arg182_1
        del arg183_1
        buf368 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf367, (1568, 512), (512, 1), 0), reinterpret_tensor(arg184_1, (512, 1536), (1, 512), 0), out=buf368)
        del arg184_1
        buf369 = reinterpret_tensor(buf367, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [x_557], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf368, arg185_1, buf369, 802816, grid=grid(802816), stream=stream0)
        buf370 = reinterpret_tensor(buf355, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf355  # reuse
        # Topologically Sorted Source Nodes: [x_557], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf368, arg185_1, buf370, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf371 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [x_557], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf370, (128, 32, 196), (6272, 196, 1), 0), out=buf371)
        buf375 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [x_557], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf371, buf375, 25088, 196, grid=grid(25088), stream=stream0)
        buf376 = reinterpret_tensor(buf370, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [x_557], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf368, arg185_1, buf376, 802816, grid=grid(802816), stream=stream0)
        del arg185_1
        buf377 = reinterpret_tensor(buf369, (128, 196, 32), (6272, 32, 1), 0); del buf369  # reuse
        # Topologically Sorted Source Nodes: [x_557], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf375, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf376, (128, 196, 32), (6272, 32, 1), 0), out=buf377)
        buf378 = reinterpret_tensor(buf376, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf376  # reuse
        # Topologically Sorted Source Nodes: [x_558], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf377, buf378, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf379 = reinterpret_tensor(buf377, (1568, 512), (512, 1), 0); del buf377  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf378, (1568, 512), (512, 1), 0), reinterpret_tensor(arg186_1, (512, 512), (1, 512), 0), out=buf379)
        del arg186_1
        buf383 = reinterpret_tensor(buf378, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf378  # reuse
        # Topologically Sorted Source Nodes: [x_561, x_562], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf363, buf379, arg187_1, arg188_1, arg189_1, buf383, 1568, 512, grid=grid(1568), stream=stream0)
        del arg188_1
        del arg189_1
        buf384 = reinterpret_tensor(buf361, (1568, 2048), (2048, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf383, (1568, 512), (512, 1), 0), reinterpret_tensor(arg190_1, (512, 2048), (1, 512), 0), out=buf384)
        del arg190_1
        buf385 = reinterpret_tensor(buf384, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf384  # reuse
        # Topologically Sorted Source Nodes: [x_564], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf385, arg191_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg191_1
        buf386 = reinterpret_tensor(buf383, (1568, 512), (512, 1), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf385, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg192_1, (2048, 512), (1, 2048), 0), out=buf386)
        del arg192_1
        buf387 = reinterpret_tensor(buf386, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf386  # reuse
        buf391 = reinterpret_tensor(buf339, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf339  # reuse
        # Topologically Sorted Source Nodes: [x_561, x_568, x_569], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf387, buf363, buf379, arg187_1, arg193_1, arg194_1, arg195_1, buf391, 1568, 512, grid=grid(1568), stream=stream0)
        del arg187_1
        del arg193_1
        del arg194_1
        del arg195_1
        buf392 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf391, (1568, 512), (512, 1), 0), reinterpret_tensor(arg196_1, (512, 1536), (1, 512), 0), out=buf392)
        del arg196_1
        buf393 = reinterpret_tensor(buf391, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [x_570], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf392, arg197_1, buf393, 802816, grid=grid(802816), stream=stream0)
        buf394 = reinterpret_tensor(buf379, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [x_570], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf392, arg197_1, buf394, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf395 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [x_570], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf393, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf394, (128, 32, 196), (6272, 196, 1), 0), out=buf395)
        buf399 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [x_570], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf395, buf399, 25088, 196, grid=grid(25088), stream=stream0)
        buf400 = reinterpret_tensor(buf394, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf394  # reuse
        # Topologically Sorted Source Nodes: [x_570], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf392, arg197_1, buf400, 802816, grid=grid(802816), stream=stream0)
        del arg197_1
        buf401 = reinterpret_tensor(buf393, (128, 196, 32), (6272, 32, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [x_570], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf400, (128, 196, 32), (6272, 32, 1), 0), out=buf401)
        buf402 = reinterpret_tensor(buf400, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [x_571], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf401, buf402, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf403 = reinterpret_tensor(buf401, (1568, 512), (512, 1), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf402, (1568, 512), (512, 1), 0), reinterpret_tensor(arg198_1, (512, 512), (1, 512), 0), out=buf403)
        del arg198_1
        buf407 = reinterpret_tensor(buf402, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf402  # reuse
        # Topologically Sorted Source Nodes: [x_574, x_575], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf387, buf403, arg199_1, arg200_1, arg201_1, buf407, 1568, 512, grid=grid(1568), stream=stream0)
        del arg200_1
        del arg201_1
        buf408 = reinterpret_tensor(buf385, (1568, 2048), (2048, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf407, (1568, 512), (512, 1), 0), reinterpret_tensor(arg202_1, (512, 2048), (1, 512), 0), out=buf408)
        del arg202_1
        buf409 = reinterpret_tensor(buf408, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf408  # reuse
        # Topologically Sorted Source Nodes: [x_577], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf409, arg203_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg203_1
        buf410 = reinterpret_tensor(buf407, (1568, 512), (512, 1), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf409, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg204_1, (2048, 512), (1, 2048), 0), out=buf410)
        del arg204_1
        buf411 = reinterpret_tensor(buf410, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf410  # reuse
        buf415 = reinterpret_tensor(buf363, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [x_574, x_581, x_582], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf411, buf387, buf403, arg199_1, arg205_1, arg206_1, arg207_1, buf415, 1568, 512, grid=grid(1568), stream=stream0)
        del arg199_1
        del arg205_1
        del arg206_1
        del arg207_1
        buf416 = buf392; del buf392  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf415, (1568, 512), (512, 1), 0), reinterpret_tensor(arg208_1, (512, 1536), (1, 512), 0), out=buf416)
        del arg208_1
        buf417 = reinterpret_tensor(buf415, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [x_583], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf416, arg209_1, buf417, 802816, grid=grid(802816), stream=stream0)
        buf418 = reinterpret_tensor(buf403, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [x_583], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf416, arg209_1, buf418, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf419 = buf395; del buf395  # reuse
        # Topologically Sorted Source Nodes: [x_583], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf417, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf418, (128, 32, 196), (6272, 196, 1), 0), out=buf419)
        buf423 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [x_583], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf419, buf423, 25088, 196, grid=grid(25088), stream=stream0)
        buf424 = reinterpret_tensor(buf418, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [x_583], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf416, arg209_1, buf424, 802816, grid=grid(802816), stream=stream0)
        del arg209_1
        buf425 = reinterpret_tensor(buf417, (128, 196, 32), (6272, 32, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [x_583], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf423, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf424, (128, 196, 32), (6272, 32, 1), 0), out=buf425)
        buf426 = reinterpret_tensor(buf424, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [x_584], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf425, buf426, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf427 = reinterpret_tensor(buf425, (1568, 512), (512, 1), 0); del buf425  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf426, (1568, 512), (512, 1), 0), reinterpret_tensor(arg210_1, (512, 512), (1, 512), 0), out=buf427)
        del arg210_1
        buf431 = reinterpret_tensor(buf426, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [x_587, x_588], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf411, buf427, arg211_1, arg212_1, arg213_1, buf431, 1568, 512, grid=grid(1568), stream=stream0)
        del arg212_1
        del arg213_1
        buf432 = reinterpret_tensor(buf409, (1568, 2048), (2048, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf431, (1568, 512), (512, 1), 0), reinterpret_tensor(arg214_1, (512, 2048), (1, 512), 0), out=buf432)
        del arg214_1
        buf433 = reinterpret_tensor(buf432, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf432  # reuse
        # Topologically Sorted Source Nodes: [x_590], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf433, arg215_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg215_1
        buf434 = reinterpret_tensor(buf431, (1568, 512), (512, 1), 0); del buf431  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf433, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg216_1, (2048, 512), (1, 2048), 0), out=buf434)
        del arg216_1
        buf435 = reinterpret_tensor(buf434, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf434  # reuse
        buf439 = reinterpret_tensor(buf387, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [x_587, x_594, x_595], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf435, buf411, buf427, arg211_1, arg217_1, arg218_1, arg219_1, buf439, 1568, 512, grid=grid(1568), stream=stream0)
        del arg211_1
        del arg217_1
        del arg218_1
        del arg219_1
        buf440 = buf416; del buf416  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf439, (1568, 512), (512, 1), 0), reinterpret_tensor(arg220_1, (512, 1536), (1, 512), 0), out=buf440)
        del arg220_1
        buf441 = reinterpret_tensor(buf439, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [x_596], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf440, arg221_1, buf441, 802816, grid=grid(802816), stream=stream0)
        buf442 = reinterpret_tensor(buf427, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [x_596], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf440, arg221_1, buf442, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf443 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [x_596], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf441, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf442, (128, 32, 196), (6272, 196, 1), 0), out=buf443)
        buf447 = buf423; del buf423  # reuse
        # Topologically Sorted Source Nodes: [x_596], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf443, buf447, 25088, 196, grid=grid(25088), stream=stream0)
        buf448 = reinterpret_tensor(buf442, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [x_596], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf440, arg221_1, buf448, 802816, grid=grid(802816), stream=stream0)
        del arg221_1
        buf449 = reinterpret_tensor(buf441, (128, 196, 32), (6272, 32, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [x_596], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf447, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf448, (128, 196, 32), (6272, 32, 1), 0), out=buf449)
        buf450 = reinterpret_tensor(buf448, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [x_597], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf449, buf450, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf451 = reinterpret_tensor(buf449, (1568, 512), (512, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf450, (1568, 512), (512, 1), 0), reinterpret_tensor(arg222_1, (512, 512), (1, 512), 0), out=buf451)
        del arg222_1
        buf455 = reinterpret_tensor(buf450, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf450  # reuse
        # Topologically Sorted Source Nodes: [x_600, x_601], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf435, buf451, arg223_1, arg224_1, arg225_1, buf455, 1568, 512, grid=grid(1568), stream=stream0)
        del arg224_1
        del arg225_1
        buf456 = reinterpret_tensor(buf433, (1568, 2048), (2048, 1), 0); del buf433  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf455, (1568, 512), (512, 1), 0), reinterpret_tensor(arg226_1, (512, 2048), (1, 512), 0), out=buf456)
        del arg226_1
        buf457 = reinterpret_tensor(buf456, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf456  # reuse
        # Topologically Sorted Source Nodes: [x_603], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf457, arg227_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg227_1
        buf458 = reinterpret_tensor(buf455, (1568, 512), (512, 1), 0); del buf455  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf457, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg228_1, (2048, 512), (1, 2048), 0), out=buf458)
        del arg228_1
        buf459 = reinterpret_tensor(buf458, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf458  # reuse
        buf463 = reinterpret_tensor(buf411, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [x_600, x_607, x_608], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf459, buf435, buf451, arg223_1, arg229_1, arg230_1, arg231_1, buf463, 1568, 512, grid=grid(1568), stream=stream0)
        del arg223_1
        del arg229_1
        del arg230_1
        del arg231_1
        buf464 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf463, (1568, 512), (512, 1), 0), reinterpret_tensor(arg232_1, (512, 1536), (1, 512), 0), out=buf464)
        del arg232_1
        buf465 = reinterpret_tensor(buf463, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf463  # reuse
        # Topologically Sorted Source Nodes: [x_609], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf464, arg233_1, buf465, 802816, grid=grid(802816), stream=stream0)
        buf466 = reinterpret_tensor(buf451, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [x_609], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf464, arg233_1, buf466, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf467 = buf443; del buf443  # reuse
        # Topologically Sorted Source Nodes: [x_609], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf465, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf466, (128, 32, 196), (6272, 196, 1), 0), out=buf467)
        buf471 = buf447; del buf447  # reuse
        # Topologically Sorted Source Nodes: [x_609], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf467, buf471, 25088, 196, grid=grid(25088), stream=stream0)
        buf472 = reinterpret_tensor(buf466, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [x_609], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf464, arg233_1, buf472, 802816, grid=grid(802816), stream=stream0)
        del arg233_1
        buf473 = reinterpret_tensor(buf465, (128, 196, 32), (6272, 32, 1), 0); del buf465  # reuse
        # Topologically Sorted Source Nodes: [x_609], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf471, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf472, (128, 196, 32), (6272, 32, 1), 0), out=buf473)
        buf474 = reinterpret_tensor(buf472, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf472  # reuse
        # Topologically Sorted Source Nodes: [x_610], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf473, buf474, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf475 = reinterpret_tensor(buf473, (1568, 512), (512, 1), 0); del buf473  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf474, (1568, 512), (512, 1), 0), reinterpret_tensor(arg234_1, (512, 512), (1, 512), 0), out=buf475)
        del arg234_1
        buf479 = reinterpret_tensor(buf474, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf474  # reuse
        # Topologically Sorted Source Nodes: [x_613, x_614], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf459, buf475, arg235_1, arg236_1, arg237_1, buf479, 1568, 512, grid=grid(1568), stream=stream0)
        del arg236_1
        del arg237_1
        buf480 = reinterpret_tensor(buf457, (1568, 2048), (2048, 1), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf479, (1568, 512), (512, 1), 0), reinterpret_tensor(arg238_1, (512, 2048), (1, 512), 0), out=buf480)
        del arg238_1
        buf481 = reinterpret_tensor(buf480, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [x_616], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf481, arg239_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg239_1
        buf482 = reinterpret_tensor(buf479, (1568, 512), (512, 1), 0); del buf479  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf481, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg240_1, (2048, 512), (1, 2048), 0), out=buf482)
        del arg240_1
        buf483 = reinterpret_tensor(buf482, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf482  # reuse
        buf487 = reinterpret_tensor(buf435, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf435  # reuse
        # Topologically Sorted Source Nodes: [x_613, x_620, x_621], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf483, buf459, buf475, arg235_1, arg241_1, arg242_1, arg243_1, buf487, 1568, 512, grid=grid(1568), stream=stream0)
        del arg235_1
        del arg241_1
        del arg242_1
        del arg243_1
        buf488 = buf464; del buf464  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf487, (1568, 512), (512, 1), 0), reinterpret_tensor(arg244_1, (512, 1536), (1, 512), 0), out=buf488)
        del arg244_1
        buf489 = reinterpret_tensor(buf487, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf487  # reuse
        # Topologically Sorted Source Nodes: [x_622], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf488, arg245_1, buf489, 802816, grid=grid(802816), stream=stream0)
        buf490 = reinterpret_tensor(buf475, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf475  # reuse
        # Topologically Sorted Source Nodes: [x_622], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf488, arg245_1, buf490, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf491 = buf467; del buf467  # reuse
        # Topologically Sorted Source Nodes: [x_622], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf489, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf490, (128, 32, 196), (6272, 196, 1), 0), out=buf491)
        buf495 = buf471; del buf471  # reuse
        # Topologically Sorted Source Nodes: [x_622], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf491, buf495, 25088, 196, grid=grid(25088), stream=stream0)
        buf496 = reinterpret_tensor(buf490, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [x_622], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf488, arg245_1, buf496, 802816, grid=grid(802816), stream=stream0)
        del arg245_1
        buf497 = reinterpret_tensor(buf489, (128, 196, 32), (6272, 32, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [x_622], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf495, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf496, (128, 196, 32), (6272, 32, 1), 0), out=buf497)
        buf498 = reinterpret_tensor(buf496, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf496  # reuse
        # Topologically Sorted Source Nodes: [x_623], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf497, buf498, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf499 = reinterpret_tensor(buf497, (1568, 512), (512, 1), 0); del buf497  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf498, (1568, 512), (512, 1), 0), reinterpret_tensor(arg246_1, (512, 512), (1, 512), 0), out=buf499)
        del arg246_1
        buf503 = reinterpret_tensor(buf498, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf498  # reuse
        # Topologically Sorted Source Nodes: [x_626, x_627], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf483, buf499, arg247_1, arg248_1, arg249_1, buf503, 1568, 512, grid=grid(1568), stream=stream0)
        del arg248_1
        del arg249_1
        buf504 = reinterpret_tensor(buf481, (1568, 2048), (2048, 1), 0); del buf481  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf503, (1568, 512), (512, 1), 0), reinterpret_tensor(arg250_1, (512, 2048), (1, 512), 0), out=buf504)
        del arg250_1
        buf505 = reinterpret_tensor(buf504, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf504  # reuse
        # Topologically Sorted Source Nodes: [x_629], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf505, arg251_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg251_1
        buf506 = reinterpret_tensor(buf503, (1568, 512), (512, 1), 0); del buf503  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf505, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg252_1, (2048, 512), (1, 2048), 0), out=buf506)
        del arg252_1
        buf507 = reinterpret_tensor(buf506, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf506  # reuse
        buf511 = reinterpret_tensor(buf459, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf459  # reuse
        # Topologically Sorted Source Nodes: [x_626, x_633, x_634], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf507, buf483, buf499, arg247_1, arg253_1, arg254_1, arg255_1, buf511, 1568, 512, grid=grid(1568), stream=stream0)
        del arg247_1
        del arg253_1
        del arg254_1
        del arg255_1
        buf512 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf511, (1568, 512), (512, 1), 0), reinterpret_tensor(arg256_1, (512, 1536), (1, 512), 0), out=buf512)
        del arg256_1
        buf513 = reinterpret_tensor(buf511, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [x_635], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf512, arg257_1, buf513, 802816, grid=grid(802816), stream=stream0)
        buf514 = reinterpret_tensor(buf499, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf499  # reuse
        # Topologically Sorted Source Nodes: [x_635], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf512, arg257_1, buf514, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf515 = buf491; del buf491  # reuse
        # Topologically Sorted Source Nodes: [x_635], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf513, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf514, (128, 32, 196), (6272, 196, 1), 0), out=buf515)
        buf519 = buf495; del buf495  # reuse
        # Topologically Sorted Source Nodes: [x_635], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf515, buf519, 25088, 196, grid=grid(25088), stream=stream0)
        buf520 = reinterpret_tensor(buf514, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [x_635], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf512, arg257_1, buf520, 802816, grid=grid(802816), stream=stream0)
        del arg257_1
        buf521 = reinterpret_tensor(buf513, (128, 196, 32), (6272, 32, 1), 0); del buf513  # reuse
        # Topologically Sorted Source Nodes: [x_635], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf519, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf520, (128, 196, 32), (6272, 32, 1), 0), out=buf521)
        buf522 = reinterpret_tensor(buf520, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [x_636], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf521, buf522, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf523 = reinterpret_tensor(buf521, (1568, 512), (512, 1), 0); del buf521  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf522, (1568, 512), (512, 1), 0), reinterpret_tensor(arg258_1, (512, 512), (1, 512), 0), out=buf523)
        del arg258_1
        buf527 = reinterpret_tensor(buf522, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf522  # reuse
        # Topologically Sorted Source Nodes: [x_639, x_640], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf507, buf523, arg259_1, arg260_1, arg261_1, buf527, 1568, 512, grid=grid(1568), stream=stream0)
        del arg260_1
        del arg261_1
        buf528 = reinterpret_tensor(buf505, (1568, 2048), (2048, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf527, (1568, 512), (512, 1), 0), reinterpret_tensor(arg262_1, (512, 2048), (1, 512), 0), out=buf528)
        del arg262_1
        buf529 = reinterpret_tensor(buf528, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf528  # reuse
        # Topologically Sorted Source Nodes: [x_642], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf529, arg263_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg263_1
        buf530 = reinterpret_tensor(buf527, (1568, 512), (512, 1), 0); del buf527  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf529, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg264_1, (2048, 512), (1, 2048), 0), out=buf530)
        del arg264_1
        buf531 = reinterpret_tensor(buf530, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf530  # reuse
        buf535 = reinterpret_tensor(buf483, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf483  # reuse
        # Topologically Sorted Source Nodes: [x_639, x_646, x_647], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf531, buf507, buf523, arg259_1, arg265_1, arg266_1, arg267_1, buf535, 1568, 512, grid=grid(1568), stream=stream0)
        del arg259_1
        del arg265_1
        del arg266_1
        del arg267_1
        buf536 = buf512; del buf512  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf535, (1568, 512), (512, 1), 0), reinterpret_tensor(arg268_1, (512, 1536), (1, 512), 0), out=buf536)
        del arg268_1
        buf537 = reinterpret_tensor(buf535, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf535  # reuse
        # Topologically Sorted Source Nodes: [x_648], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf536, arg269_1, buf537, 802816, grid=grid(802816), stream=stream0)
        buf538 = reinterpret_tensor(buf523, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf523  # reuse
        # Topologically Sorted Source Nodes: [x_648], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf536, arg269_1, buf538, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf539 = buf515; del buf515  # reuse
        # Topologically Sorted Source Nodes: [x_648], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf537, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf538, (128, 32, 196), (6272, 196, 1), 0), out=buf539)
        buf543 = buf519; del buf519  # reuse
        # Topologically Sorted Source Nodes: [x_648], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf539, buf543, 25088, 196, grid=grid(25088), stream=stream0)
        buf544 = reinterpret_tensor(buf538, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf538  # reuse
        # Topologically Sorted Source Nodes: [x_648], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf536, arg269_1, buf544, 802816, grid=grid(802816), stream=stream0)
        del arg269_1
        buf545 = reinterpret_tensor(buf537, (128, 196, 32), (6272, 32, 1), 0); del buf537  # reuse
        # Topologically Sorted Source Nodes: [x_648], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf543, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf544, (128, 196, 32), (6272, 32, 1), 0), out=buf545)
        buf546 = reinterpret_tensor(buf544, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf544  # reuse
        # Topologically Sorted Source Nodes: [x_649], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf545, buf546, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf547 = reinterpret_tensor(buf545, (1568, 512), (512, 1), 0); del buf545  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf546, (1568, 512), (512, 1), 0), reinterpret_tensor(arg270_1, (512, 512), (1, 512), 0), out=buf547)
        del arg270_1
        buf551 = reinterpret_tensor(buf546, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf546  # reuse
        # Topologically Sorted Source Nodes: [x_652, x_653], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf531, buf547, arg271_1, arg272_1, arg273_1, buf551, 1568, 512, grid=grid(1568), stream=stream0)
        del arg272_1
        del arg273_1
        buf552 = reinterpret_tensor(buf529, (1568, 2048), (2048, 1), 0); del buf529  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf551, (1568, 512), (512, 1), 0), reinterpret_tensor(arg274_1, (512, 2048), (1, 512), 0), out=buf552)
        del arg274_1
        buf553 = reinterpret_tensor(buf552, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf552  # reuse
        # Topologically Sorted Source Nodes: [x_655], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf553, arg275_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg275_1
        buf554 = reinterpret_tensor(buf551, (1568, 512), (512, 1), 0); del buf551  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf553, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg276_1, (2048, 512), (1, 2048), 0), out=buf554)
        del arg276_1
        buf555 = reinterpret_tensor(buf554, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf554  # reuse
        buf559 = reinterpret_tensor(buf507, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf507  # reuse
        # Topologically Sorted Source Nodes: [x_652, x_659, x_660], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf555, buf531, buf547, arg271_1, arg277_1, arg278_1, arg279_1, buf559, 1568, 512, grid=grid(1568), stream=stream0)
        del arg271_1
        del arg277_1
        del arg278_1
        del arg279_1
        buf560 = buf536; del buf536  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf559, (1568, 512), (512, 1), 0), reinterpret_tensor(arg280_1, (512, 1536), (1, 512), 0), out=buf560)
        del arg280_1
        buf561 = reinterpret_tensor(buf559, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf559  # reuse
        # Topologically Sorted Source Nodes: [x_661], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf560, arg281_1, buf561, 802816, grid=grid(802816), stream=stream0)
        buf562 = reinterpret_tensor(buf547, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [x_661], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf560, arg281_1, buf562, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf563 = buf539; del buf539  # reuse
        # Topologically Sorted Source Nodes: [x_661], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf561, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf562, (128, 32, 196), (6272, 196, 1), 0), out=buf563)
        buf567 = buf543; del buf543  # reuse
        # Topologically Sorted Source Nodes: [x_661], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf563, buf567, 25088, 196, grid=grid(25088), stream=stream0)
        buf568 = reinterpret_tensor(buf562, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf562  # reuse
        # Topologically Sorted Source Nodes: [x_661], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf560, arg281_1, buf568, 802816, grid=grid(802816), stream=stream0)
        del arg281_1
        buf569 = reinterpret_tensor(buf561, (128, 196, 32), (6272, 32, 1), 0); del buf561  # reuse
        # Topologically Sorted Source Nodes: [x_661], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf567, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf568, (128, 196, 32), (6272, 32, 1), 0), out=buf569)
        buf570 = reinterpret_tensor(buf568, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf568  # reuse
        # Topologically Sorted Source Nodes: [x_662], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf569, buf570, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf571 = reinterpret_tensor(buf569, (1568, 512), (512, 1), 0); del buf569  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf570, (1568, 512), (512, 1), 0), reinterpret_tensor(arg282_1, (512, 512), (1, 512), 0), out=buf571)
        del arg282_1
        buf575 = reinterpret_tensor(buf570, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf570  # reuse
        # Topologically Sorted Source Nodes: [x_665, x_666], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf555, buf571, arg283_1, arg284_1, arg285_1, buf575, 1568, 512, grid=grid(1568), stream=stream0)
        del arg284_1
        del arg285_1
        buf576 = reinterpret_tensor(buf553, (1568, 2048), (2048, 1), 0); del buf553  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf575, (1568, 512), (512, 1), 0), reinterpret_tensor(arg286_1, (512, 2048), (1, 512), 0), out=buf576)
        del arg286_1
        buf577 = reinterpret_tensor(buf576, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf576  # reuse
        # Topologically Sorted Source Nodes: [x_668], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf577, arg287_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg287_1
        buf578 = reinterpret_tensor(buf575, (1568, 512), (512, 1), 0); del buf575  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf577, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg288_1, (2048, 512), (1, 2048), 0), out=buf578)
        del arg288_1
        buf579 = reinterpret_tensor(buf578, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf578  # reuse
        buf583 = reinterpret_tensor(buf531, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf531  # reuse
        # Topologically Sorted Source Nodes: [x_665, x_672, x_673], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf579, buf555, buf571, arg283_1, arg289_1, arg290_1, arg291_1, buf583, 1568, 512, grid=grid(1568), stream=stream0)
        del arg283_1
        del arg289_1
        del arg290_1
        del arg291_1
        del buf555
        buf584 = buf560; del buf560  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf583, (1568, 512), (512, 1), 0), reinterpret_tensor(arg292_1, (512, 1536), (1, 512), 0), out=buf584)
        del arg292_1
        buf585 = reinterpret_tensor(buf583, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [x_674], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_36.run(buf584, arg293_1, buf585, 802816, grid=grid(802816), stream=stream0)
        buf586 = reinterpret_tensor(buf571, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [x_674], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_37.run(buf584, arg293_1, buf586, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf587 = buf563; del buf563  # reuse
        # Topologically Sorted Source Nodes: [x_674], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf585, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf586, (128, 32, 196), (6272, 196, 1), 0), out=buf587)
        buf591 = buf567; del buf567  # reuse
        # Topologically Sorted Source Nodes: [x_674], Original ATen: [aten._safe_softmax]
        triton_red_fused__safe_softmax_38.run(buf587, buf591, 25088, 196, grid=grid(25088), stream=stream0)
        del buf587
        buf592 = reinterpret_tensor(buf586, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf586  # reuse
        # Topologically Sorted Source Nodes: [x_674], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf584, arg293_1, buf592, 802816, grid=grid(802816), stream=stream0)
        del arg293_1
        del buf584
        buf593 = reinterpret_tensor(buf585, (128, 196, 32), (6272, 32, 1), 0); del buf585  # reuse
        # Topologically Sorted Source Nodes: [x_674], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf591, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf592, (128, 196, 32), (6272, 32, 1), 0), out=buf593)
        del buf591
        buf594 = reinterpret_tensor(buf592, (8, 1, 196, 32, 16), (100352, 1, 512, 16, 1), 0); del buf592  # reuse
        # Topologically Sorted Source Nodes: [x_675], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf593, buf594, 50176, 16, grid=grid(50176, 16), stream=stream0)
        buf595 = reinterpret_tensor(buf593, (1568, 512), (512, 1), 0); del buf593  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf594, (1568, 512), (512, 1), 0), reinterpret_tensor(arg294_1, (512, 512), (1, 512), 0), out=buf595)
        del arg294_1
        buf599 = reinterpret_tensor(buf594, (8, 1, 196, 512), (100352, 1, 512, 1), 0); del buf594  # reuse
        # Topologically Sorted Source Nodes: [x_678, x_679], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_45.run(buf579, buf595, arg295_1, arg296_1, arg297_1, buf599, 1568, 512, grid=grid(1568), stream=stream0)
        del arg296_1
        del arg297_1
        buf600 = reinterpret_tensor(buf577, (1568, 2048), (2048, 1), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf599, (1568, 512), (512, 1), 0), reinterpret_tensor(arg298_1, (512, 2048), (1, 512), 0), out=buf600)
        del arg298_1
        buf601 = reinterpret_tensor(buf600, (8, 1, 196, 2048), (401408, 1, 2048, 1), 0); del buf600  # reuse
        # Topologically Sorted Source Nodes: [x_681], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_43.run(buf601, arg299_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg299_1
        buf602 = reinterpret_tensor(buf599, (1568, 512), (512, 1), 0); del buf599  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf601, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg300_1, (2048, 512), (1, 2048), 0), out=buf602)
        del arg300_1
        del buf601
        buf603 = reinterpret_tensor(buf602, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf602  # reuse
        buf604 = reinterpret_tensor(buf141, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf141  # reuse
        buf605 = reinterpret_tensor(buf140, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_678, x_685, x_688], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf603, buf579, buf595, arg295_1, arg301_1, buf604, buf605, 1568, 512, grid=grid(1568), stream=stream0)
        del arg295_1
        del arg301_1
        del buf579
        del buf595
        buf607 = empty_strided_cuda((8, 512, 1, 1, 2), (1024, 1, 8192, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_690], Original ATen: [aten.mean]
        triton_red_fused_mean_48.run(buf603, buf604, buf605, arg302_1, arg303_1, buf607, 8192, 98, grid=grid(8192), stream=stream0)
        del arg302_1
        del arg303_1
        del buf603
        del buf604
        del buf605
        buf609 = empty_strided_cuda((8, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_690], Original ATen: [aten.mean]
        triton_per_fused_mean_49.run(buf607, buf609, 4096, 2, grid=grid(4096), stream=stream0)
        del buf607
        buf610 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_693], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg305_1, reinterpret_tensor(buf609, (8, 512), (512, 1), 0), reinterpret_tensor(arg304_1, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf610)
        del arg304_1
        del arg305_1
        del buf609
    return (buf610, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 16, 196, 128), (401408, 25088, 128, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((1, 4, 196, 256), (200704, 50176, 256, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('jx_nest_base', benchmark_compiled_module)
