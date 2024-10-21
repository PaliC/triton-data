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


# kernel path: /tmp/torchinductor_sahanp/uz/cuzvt3uvmzuoklqmhk5o22ige4lh5taikg4ohr77h5ay4o5mcqqn.py
# Topologically Sorted Source Nodes: [layer_norm_63], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_63 => var_mean_63
# Graph fragment:
#   %var_mean_63 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_501, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_red_fused_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((4*(x1 % 14)) + (56*((r3 + (128*x0)) // 96)) + (56*((((r3 + (128*x0)) // 24) % 4) // 4)) + (224*(x1 // 14)) + (3136*(triton_helpers.div_floor_integer((4*((r3 + (128*x0)) // 96)) + (((r3 + (128*x0)) // 24) % 4),  16))) + (3136*((r3 + (128*x0)) % 24)) + (75264*x2) + (((r3 + (128*x0)) // 24) % 4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((triton_helpers.div_floor_integer((4*((r3 + (128*x0)) // 96)) + (((r3 + (128*x0)) // 24) % 4),  16)) + ((r3 + (128*x0)) % 24)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((16*((r3 + (128*x0)) % 24)) + ((r3 + (128*x0)) // 24)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x5), tmp6, xmask)
    tl.store(out_ptr1 + (x5), tmp7, xmask)
    tl.store(out_ptr2 + (x5), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xh/cxhkqzvvapbze6m4jytfrbdj5i52hdgaxxytnl33cf7f2onikv4j.py
# Topologically Sorted Source Nodes: [layer_norm_63], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_63 => var_mean_63
# Graph fragment:
#   %var_mean_63 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_501, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_1 = async_compile.triton('triton_per_fused_native_layer_norm_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (3*x0)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ia/cia47grwrzfqbfngpvjoj47w43yckw7cv7a6if77st75tib75b2j.py
# Topologically Sorted Source Nodes: [layer_norm_63], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_63 => add_217, add_218, mul_222, mul_223, rsqrt_63, sub_87, var_mean_63
# Graph fragment:
#   %var_mean_63 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_501, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_501, %getitem_175), kwargs = {})
#   %add_217 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_174, 1e-05), kwargs = {})
#   %rsqrt_63 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_217,), kwargs = {})
#   %mul_222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %rsqrt_63), kwargs = {})
#   %mul_223 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_222, %arg4_1), kwargs = {})
#   %add_218 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_223, %arg5_1), kwargs = {})
triton_poi_fused_native_layer_norm_2 = async_compile.triton('triton_poi_fused_native_layer_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 384
    x1 = (xindex // 384) % 196
    x2 = (xindex // 75264)
    x3 = (xindex // 384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((4*(x1 % 14)) + (56*(x0 // 96)) + (56*(((x0 // 24) % 4) // 4)) + (224*(x1 // 14)) + (3136*(triton_helpers.div_floor_integer((4*(x0 // 96)) + ((x0 // 24) % 4),  16))) + (3136*(x0 % 24)) + (75264*x2) + ((x0 // 24) % 4)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((triton_helpers.div_floor_integer((4*(x0 // 96)) + ((x0 // 24) % 4),  16)) + (x0 % 24)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((16*(x0 % 24)) + (x0 // 24)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 384.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xb/cxbojvezkp4jg7islbta7fvyiwgfnld5aicbztzraxootfwdpqt7.py
# Topologically Sorted Source Nodes: [patch_embed_41], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   patch_embed_41 => var_mean_64
# Graph fragment:
#   %var_mean_64 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_503, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_3 = async_compile.triton('triton_per_fused_native_layer_norm_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_3(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 384, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qi/cqiisf677s5xo3gq4zogt3lm3uwgsrjyt4kn7q6eqr4xgixskaof.py
# Topologically Sorted Source Nodes: [layer_norm_65], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_65 => clone_189, var_mean_65
# Graph fragment:
#   %clone_189 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_235,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_65 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_189, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_4 = async_compile.triton('triton_per_fused_native_layer_norm_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((4*((x1 % 196) % 14)) + (56*(x0 // 4)) + (56*((x0 % 4) // 4)) + (224*((x1 % 196) // 14)) + (3136*r2) + (3136*(triton_helpers.div_floor_integer((4*(x0 // 4)) + (x0 % 4),  16))) + (75264*(x1 // 196)) + (x0 % 4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (triton_helpers.div_floor_integer((4*(x0 // 4)) + (x0 % 4),  16))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (16*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/y4/cy4fgwax2jocv2rtxnozfu3cqsjlll4kxw5oobbdufu257cbcwsu.py
# Topologically Sorted Source Nodes: [layer_norm_65], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_65 => add_222, add_223, clone_189, mul_226, mul_227, rsqrt_65, sub_89, var_mean_65
# Graph fragment:
#   %clone_189 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_235,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_65 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_189, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_189, %getitem_179), kwargs = {})
#   %add_222 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_178, 1e-05), kwargs = {})
#   %rsqrt_65 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_222,), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %rsqrt_65), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_226, %arg12_1), kwargs = {})
#   %add_223 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_227, %arg13_1), kwargs = {})
triton_poi_fused_native_layer_norm_5 = async_compile.triton('triton_poi_fused_native_layer_norm_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 37632
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + ((4*((y1 % 196) % 14)) + (56*(x2 // 4)) + (56*((x2 % 4) // 4)) + (224*((y1 % 196) // 14)) + (3136*y0) + (3136*(triton_helpers.div_floor_integer((4*(x2 // 4)) + (x2 % 4),  16))) + (75264*(y1 // 196)) + (x2 % 4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (triton_helpers.div_floor_integer((4*(x2 // 4)) + (x2 % 4),  16))), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (16*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (16*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (16*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 24.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (y0 + (24*x2) + (384*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lh/clhwqep6uzlbitxon7dglbsrs6lkisu32y4rpc22np53m2lhzmcw.py
# Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_48 => clone_190
# Graph fragment:
#   %clone_190 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_98,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 6
    x1 = (xindex // 6) % 16
    x2 = (xindex // 96) % 4
    x3 = (xindex // 384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (6*x2) + (48*x1) + (768*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/f4/cf4tt2hjjyscbygass2yl6hamew2biwwbwzb463lntqqmq37stu2.py
# Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_48 => clone_191
# Graph fragment:
#   %clone_191 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_99,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 37632
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (24 + y0 + (48*x2) + (768*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ik/cikgyqkib36anh7kuvp6pmvd6yjnalurpebfqlvfqsvizxlasaxp.py
# Topologically Sorted Source Nodes: [attn_73], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_73 => div_24, exp_24, sum_25
# Graph fragment:
#   %mul_tensor_46 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_512, 1), kwargs = {})
#   %amax_default_23 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_46, [-1], True), kwargs = {})
#   %sub_tensor_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_46, %amax_default_23), kwargs = {})
#   %mul_tensor_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_23, 0.408248290463863), kwargs = {})
#   %exp_24 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_47,), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [-1], True), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_24, %sum_25), kwargs = {})
triton_per_fused__softmax_8 = async_compile.triton('triton_per_fused__softmax_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_8(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), xmask, other=0.0)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = 0.408248290463863
    tmp9 = tmp7 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r1 + (16*x0)), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pb/cpbnoignsnutdievbqkwscnavnkk3zilwxhhpkarryteifvwnx7i.py
# Topologically Sorted Source Nodes: [matmul_49], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_49 => clone_192
# Graph fragment:
#   %clone_192 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_101,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_poi_fused_clone_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 6
    x1 = (xindex // 6) % 16
    x2 = (xindex // 96) % 4
    x3 = (xindex // 384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (6*x2) + (24*x1) + (384*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/c5/cc5mibseaqpa2k6wdvqr5tnrndkuxtv6g66kqiyct6isiiufcyr5.py
# Topologically Sorted Source Nodes: [x_205], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_205 => clone_193
# Graph fragment:
#   %clone_193 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_242,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_10 = async_compile.triton('triton_poi_fused_clone_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 6
    x1 = (xindex // 6) % 4
    x2 = (xindex // 24) % 16
    x3 = (xindex // 384)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (6*x2) + (96*x1) + (384*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5n/c5nvsqx6aw2d5l2hz2vcqrsi7lx3w6upahqed4cpjyoqpgddoapd.py
# Topologically Sorted Source Nodes: [pixel_embed_24, layer_norm_66], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_66 => add_225, add_226, clone_194, mul_229, mul_230, rsqrt_66, sub_91, var_mean_66
#   pixel_embed_24 => add_224
# Graph fragment:
#   %add_224 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_235, %view_518), kwargs = {})
#   %clone_194 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_224,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_66 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_194, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_194, %getitem_183), kwargs = {})
#   %add_225 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_182, 1e-05), kwargs = {})
#   %rsqrt_66 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_225,), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %rsqrt_66), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_229, %arg18_1), kwargs = {})
#   %add_226 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_230, %arg19_1), kwargs = {})
triton_per_fused_add_native_layer_norm_11 = async_compile.triton('triton_per_fused_add_native_layer_norm_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((4*((x1 % 196) % 14)) + (56*(x0 // 4)) + (56*((x0 % 4) // 4)) + (224*((x1 % 196) // 14)) + (3136*r2) + (3136*(triton_helpers.div_floor_integer((4*(x0 // 4)) + (x0 % 4),  16))) + (75264*(x1 // 196)) + (x0 % 4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (triton_helpers.div_floor_integer((4*(x0 // 4)) + (x0 % 4),  16))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (16*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (24*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 24.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (24*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (24*x3)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kl/cklh6lfc3mj4o7u3js4fudnxi5q6lvgzyltgp7j6eabn7r6ekqo7.py
# Topologically Sorted Source Nodes: [x_209], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_209 => add_227, erf_24, mul_231, mul_232, mul_233
# Graph fragment:
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_520, 0.5), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_520, 0.7071067811865476), kwargs = {})
#   %erf_24 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_232,), kwargs = {})
#   %add_227 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_24, 1), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_231, %add_227), kwargs = {})
triton_poi_fused_gelu_12 = async_compile.triton('triton_poi_fused_gelu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 96
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


# kernel path: /tmp/torchinductor_sahanp/tg/ctgyeu4csbp3jcbrhp2cbyn5f4tgkfertslgqccp5lp7zrytcrqe.py
# Topologically Sorted Source Nodes: [pixel_embed_25, layer_norm_67, layer_norm_70], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_67 => add_229, add_230, clone_197, mul_234, mul_235, rsqrt_67, sub_92, var_mean_67
#   layer_norm_70 => add_239, add_240, clone_204, mul_244, mul_245, rsqrt_70, sub_96, var_mean_70
#   pixel_embed_25 => add_228
# Graph fragment:
#   %add_228 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_224, %view_522), kwargs = {})
#   %clone_197 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_228,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_67 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_197, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_197, %getitem_185), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_184, 1e-05), kwargs = {})
#   %rsqrt_67 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_229,), kwargs = {})
#   %mul_234 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %rsqrt_67), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_234, %arg24_1), kwargs = {})
#   %add_230 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_235, %arg25_1), kwargs = {})
#   %clone_204 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_228,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_70 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_204, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_204, %getitem_193), kwargs = {})
#   %add_239 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_192, 1e-05), kwargs = {})
#   %rsqrt_70 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_239,), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %rsqrt_70), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_244, %arg40_1), kwargs = {})
#   %add_240 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_245, %arg41_1), kwargs = {})
triton_per_fused_add_native_layer_norm_13 = async_compile.triton('triton_per_fused_add_native_layer_norm_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 6, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 24.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(out_ptr4 + (r1 + (24*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (24*x0)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qq/cqqi3n32d2qplnvbt2swczefxh3bcrgackcodacpr5ndhebtmvtc.py
# Topologically Sorted Source Nodes: [patch_embed_42, patch_embed_43], Original ATen: [aten.cat, aten.add]
# Source node to ATen node mapping:
#   patch_embed_42 => cat_13
#   patch_embed_43 => add_221
# Graph fragment:
#   %cat_13 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_97, %add_220], 1), kwargs = {})
#   %add_221 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_13, %arg11_1), kwargs = {})
triton_poi_fused_add_cat_14 = async_compile.triton('triton_poi_fused_add_cat_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 384) % 197
    x0 = xindex % 384
    x2 = (xindex // 75648)
    x3 = xindex % 75648
    x4 = xindex
    tmp26 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + (384*((-1) + x1)) + (75264*x2)), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + ((196*x2) + ((-1) + x1)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((196*x2) + ((-1) + x1)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 384.0
    tmp14 = tmp12 / tmp13
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 * tmp19
    tmp21 = tl.load(in_ptr5 + (x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp6, tmp22, tmp23)
    tmp25 = tl.where(tmp4, tmp5, tmp24)
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr0 + (x4), tmp27, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jd/cjdip27pl6elsramm66s7gb7rx4jyfilfpltc5tpkwyhucwwso5m.py
# Topologically Sorted Source Nodes: [patch_embed_45, layer_norm_68], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_68 => add_232, add_233, mul_236, mul_237, rsqrt_68, sub_93, var_mean_68
#   patch_embed_45 => cat_14
# Graph fragment:
#   %cat_14 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_55, %add_231], 1), kwargs = {})
#   %var_mean_68 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_14, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_14, %getitem_187), kwargs = {})
#   %add_232 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_186, 1e-05), kwargs = {})
#   %rsqrt_68 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_232,), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %rsqrt_68), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_236, %arg28_1), kwargs = {})
#   %add_233 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_237, %arg29_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_15 = async_compile.triton('triton_per_fused_cat_native_layer_norm_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
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
    tmp5 = tl.load(in_ptr0 + (r2 + (75648*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (384 + r2 + (384*((-1) + x0)) + (75648*x1)), rmask & tmp6, other=0.0)
    tmp10 = tl.load(in_ptr1 + (r2 + (384*((-1) + x0)) + (75264*x1)), rmask & tmp6, other=0.0)
    tmp11 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 384, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 384.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-05
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr3 + (r2 + (384*x3)), tmp43, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/i2/ci2pqw3v4ify2bjv6pmvlqgrhiiljqmqc4auo6pm62k4ekrbv6jg.py
# Topologically Sorted Source Nodes: [matmul_50], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_50 => clone_198
# Graph fragment:
#   %clone_198 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_102,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_16 = async_compile.triton('triton_poi_fused_clone_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 197
    x2 = (xindex // 12608) % 6
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (151296*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rw/crwmk57fyzqrolcdaukbq44ozzbvmsqogofqrbvd3m3halxreur3.py
# Topologically Sorted Source Nodes: [matmul_50], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_50 => clone_199
# Graph fragment:
#   %clone_199 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_103,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_17 = async_compile.triton('triton_poi_fused_clone_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 197
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (384 + y0 + (768*x2) + (151296*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (197*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ky/cky2medaff5a4ze62d453eworfo367jcsmpzqbqqgehyyfzoqkwa.py
# Topologically Sorted Source Nodes: [attn_76], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_76 => div_25, exp_25, sum_26
# Graph fragment:
#   %mul_tensor_44 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_534, 1), kwargs = {})
#   %amax_default_22 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_44, [-1], True), kwargs = {})
#   %sub_tensor_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_44, %amax_default_22), kwargs = {})
#   %mul_tensor_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_22, 0.125), kwargs = {})
#   %exp_25 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_45,), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_25, [-1], True), kwargs = {})
#   %div_25 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_25, %sum_26), kwargs = {})
triton_red_fused__softmax_18 = async_compile.triton('triton_red_fused__softmax_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_18(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9456
    rnumel = 197
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x5 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x0 = xindex % 1182
    x1 = (xindex // 1182)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (197*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 1.0
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp6 = tl.load(in_ptr0 + (r2 + (197*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8 - tmp4
        tmp10 = 0.125
        tmp11 = tmp9 * tmp10
        tmp12 = tl_math.exp(tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    x3 = xindex % 197
    x6 = (xindex // 197)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp16 = tl.load(in_ptr0 + (r2 + (197*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = 1.0
        tmp18 = tmp16 * tmp17
        tmp19 = tmp18 - tmp4
        tmp20 = 0.125
        tmp21 = tmp19 * tmp20
        tmp22 = tl_math.exp(tmp21)
        tmp23 = tmp22 / tmp14
        tl.store(out_ptr2 + (r2 + (197*x3) + (38816*x6)), tmp23, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ea/ceafsi6uqzrtc7zutu5k6caqhs4sqr3vydii6nev3wrjtxam3qey.py
# Topologically Sorted Source Nodes: [matmul_51], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_51 => clone_200
# Graph fragment:
#   %clone_200 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_105,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_19 = async_compile.triton('triton_poi_fused_clone_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 197
    x2 = (xindex // 12608) % 6
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (384*x1) + (75648*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/td/ctdj3i2eg3quddyfgvxoslj2xptqltllzqmtojdbwt4l7wuh6vz6.py
# Topologically Sorted Source Nodes: [x_213], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_213 => clone_201
# Graph fragment:
#   %clone_201 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_252,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_20 = async_compile.triton('triton_poi_fused_clone_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 6
    x2 = (xindex // 384) % 197
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (12608*x1) + (75648*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zy/czytwhcyx7t27d5xi76un74gu4fzdbor5z3vqxggbghfvfxcuyrh.py
# Topologically Sorted Source Nodes: [patch_embed_45, patch_embed_46, layer_norm_69], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_69 => add_235, add_236, mul_239, mul_240, rsqrt_69, sub_95, var_mean_69
#   patch_embed_45 => cat_14
#   patch_embed_46 => add_234
# Graph fragment:
#   %cat_14 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_55, %add_231], 1), kwargs = {})
#   %add_234 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_14, %view_540), kwargs = {})
#   %var_mean_69 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_234, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_234, %getitem_191), kwargs = {})
#   %add_235 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_190, 1e-05), kwargs = {})
#   %rsqrt_69 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_235,), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %rsqrt_69), kwargs = {})
#   %mul_240 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_239, %arg34_1), kwargs = {})
#   %add_236 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_240, %arg35_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_21 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
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
    tmp17 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (75648*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (384 + r2 + (384*((-1) + x0)) + (75648*x1)), rmask & tmp6, other=0.0)
    tmp10 = tl.load(in_ptr1 + (r2 + (384*((-1) + x0)) + (75264*x1)), rmask & tmp6, other=0.0)
    tmp11 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
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
    tmp28 = tl.full([1], 384, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = tmp20 - tmp30
    tmp38 = 384.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-05
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(in_out_ptr0 + (r2 + (384*x3)), tmp20, rmask)
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp47, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/c7/cc7qlw4psh7x6hsk3ixxdpkjgx5726tpfluwnlla4zskntwja7wk.py
# Topologically Sorted Source Nodes: [pixel_embed_25, pixel_embed_26, layer_norm_71], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_71 => add_242, add_243, clone_209, mul_247, mul_248, rsqrt_71, sub_98, var_mean_71
#   pixel_embed_25 => add_228
#   pixel_embed_26 => add_241
# Graph fragment:
#   %add_228 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_224, %view_522), kwargs = {})
#   %add_241 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_228, %view_559), kwargs = {})
#   %clone_209 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_241,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_71 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_209, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_209, %getitem_197), kwargs = {})
#   %add_242 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_196, 1e-05), kwargs = {})
#   %rsqrt_71 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_242,), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_98, %rsqrt_71), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_247, %arg46_1), kwargs = {})
#   %add_243 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_248, %arg47_1), kwargs = {})
triton_per_fused_add_native_layer_norm_22 = async_compile.triton('triton_per_fused_add_native_layer_norm_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 24.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (24*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (24*x0)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/he/chewh2qyunplcx66d3pa3zny4x6x3wbwycqyog4crfkdiwjlfcxc.py
# Topologically Sorted Source Nodes: [x_217], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_217 => add_237, erf_25, mul_241, mul_242, mul_243
# Graph fragment:
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_542, 0.5), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_542, 0.7071067811865476), kwargs = {})
#   %erf_25 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_242,), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_25, 1), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_241, %add_237), kwargs = {})
triton_poi_fused_gelu_23 = async_compile.triton('triton_poi_fused_gelu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_23(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2420736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1536
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


# kernel path: /tmp/torchinductor_sahanp/eu/ceuagrcpz6rvvlzqd4kgkmhsamuauozoomwfxwkhwfv7ncxopyxr.py
# Topologically Sorted Source Nodes: [patch_embed_48, layer_norm_73], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_73 => add_249, add_250, mul_254, mul_255, rsqrt_73, sub_100, var_mean_73
#   patch_embed_48 => cat_15
# Graph fragment:
#   %cat_15 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_59, %add_248], 1), kwargs = {})
#   %var_mean_73 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_15, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_15, %getitem_201), kwargs = {})
#   %add_249 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_200, 1e-05), kwargs = {})
#   %rsqrt_73 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_249,), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %rsqrt_73), kwargs = {})
#   %mul_255 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_254, %arg56_1), kwargs = {})
#   %add_250 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_255, %arg57_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_24 = async_compile.triton('triton_per_fused_cat_native_layer_norm_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 10, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
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
    tmp50 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (384*x0) + (75648*x1)), rmask & tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (r2 + (384*x0) + (75648*x1)), rmask & tmp4, other=0.0)
    tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 197, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr0 + (384 + r2 + (384*((-1) + x0)) + (75648*x1)), rmask & tmp12, other=0.0)
    tmp16 = tl.load(in_ptr1 + (384 + r2 + (384*((-1) + x0)) + (75648*x1)), rmask & tmp12, other=0.0)
    tmp17 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp12, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.load(in_ptr3 + (r2 + (384*((-1) + x0)) + (75264*x1)), rmask & tmp12, other=0.0)
    tmp21 = tl.load(in_ptr4 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp12, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp19 + tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp12, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp11, tmp25)
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask, tmp27, 0)
    tmp30 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp32 = tl.where(rmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = tl.full([1], 384, tl.int32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 / tmp35
    tmp37 = tmp27 - tmp36
    tmp38 = tmp37 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [RBLOCK])
    tmp41 = tl.where(rmask, tmp39, 0)
    tmp42 = triton_helpers.promote_to_tensor(tl.sum(tmp41, 0))
    tmp43 = tmp26 - tmp36
    tmp44 = 384.0
    tmp45 = tmp42 / tmp44
    tmp46 = 1e-05
    tmp47 = tmp45 + tmp46
    tmp48 = libdevice.rsqrt(tmp47)
    tmp49 = tmp43 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tl.store(out_ptr0 + (r2 + (384*x3)), tmp26, rmask)
    tl.store(out_ptr3 + (r2 + (384*x3)), tmp53, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jt/cjtgxlbkin3u4wnldgfj3dabvsfa4dhp7rf4swgzu7cqs26kiegx.py
# Topologically Sorted Source Nodes: [patch_embed_49, layer_norm_74], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_74 => add_252, add_253, mul_257, mul_258, rsqrt_74, sub_102, var_mean_74
#   patch_embed_49 => add_251
# Graph fragment:
#   %add_251 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_15, %view_581), kwargs = {})
#   %var_mean_74 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_251, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_251, %getitem_205), kwargs = {})
#   %add_252 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_204, 1e-05), kwargs = {})
#   %rsqrt_74 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_252,), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %rsqrt_74), kwargs = {})
#   %mul_258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_257, %arg62_1), kwargs = {})
#   %add_253 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_258, %arg63_1), kwargs = {})
triton_per_fused_add_native_layer_norm_25 = async_compile.triton('triton_per_fused_add_native_layer_norm_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask, other=0.0)
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
    tmp12 = tl.full([1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 384.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/s5/cs5wbk7hrgcdy7dpg2cor576cgvurzp46q26l5xjfcxv4akoc6cd.py
# Topologically Sorted Source Nodes: [patch_embed_49, patch_embed_50], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   patch_embed_49 => add_251
#   patch_embed_50 => add_255
# Graph fragment:
#   %add_251 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_15, %view_581), kwargs = {})
#   %add_255 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_251, %view_585), kwargs = {})
triton_poi_fused_add_26 = async_compile.triton('triton_poi_fused_add_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wf/cwfx4wwjzycjiwkxbix3gvqo4z5aynx4dmntue6bsuwk5ojfm4oa.py
# Topologically Sorted Source Nodes: [pixel_embed_47, layer_norm_122], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_122 => add_416, add_417, clone_362, mul_432, mul_433, rsqrt_122, sub_169, var_mean_122
#   pixel_embed_47 => add_415
# Graph fragment:
#   %add_415 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_411, %view_973), kwargs = {})
#   %clone_362 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_415,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_122 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_362, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_169 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_362, %getitem_339), kwargs = {})
#   %add_416 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_338, 1e-05), kwargs = {})
#   %rsqrt_122 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_416,), kwargs = {})
#   %mul_432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_169, %rsqrt_122), kwargs = {})
#   %mul_433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_432, %arg332_1), kwargs = {})
#   %add_417 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_433, %arg333_1), kwargs = {})
triton_per_fused_add_native_layer_norm_27 = async_compile.triton('triton_per_fused_add_native_layer_norm_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 24.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (24*x0)), tmp31, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oh/cohlhav3zzhvfv4mmtawsn4vrwewmzqvtrhpj47dqh6zw6gx6jya.py
# Topologically Sorted Source Nodes: [patch_embed_79, patch_embed_80, patch_embed_81], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   patch_embed_79 => add_421
#   patch_embed_80 => add_425
#   patch_embed_81 => var_mean_125
# Graph fragment:
#   %add_421 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_25, %view_991), kwargs = {})
#   %add_425 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_421, %view_995), kwargs = {})
#   %var_mean_125 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_425, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_28 = async_compile.triton('triton_per_fused_add_native_layer_norm_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp8, rmask)
    tl.store(out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (x0), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gi/cgii4d6sxobfuopgg6irux6ynhjwj67ookmkrslt57sdwfprrmq2.py
# Topologically Sorted Source Nodes: [x_398], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_398 => clone_369
# Graph fragment:
#   %clone_369 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%select_1,), kwargs = {})
triton_poi_fused_clone_29 = async_compile.triton('triton_poi_fused_clone_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (75648*x1)), xmask)
    tmp1 = tl.load(in_ptr1 + (197*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (197*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 384.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (1, 24, 4, 4), (384, 16, 4, 1))
    assert_size_stride(arg2_1, (24, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg3_1, (24, ), (1, ))
    assert_size_stride(arg4_1, (384, ), (1, ))
    assert_size_stride(arg5_1, (384, ), (1, ))
    assert_size_stride(arg6_1, (384, 384), (384, 1))
    assert_size_stride(arg7_1, (384, ), (1, ))
    assert_size_stride(arg8_1, (384, ), (1, ))
    assert_size_stride(arg9_1, (384, ), (1, ))
    assert_size_stride(arg10_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg11_1, (1, 197, 384), (75648, 384, 1))
    assert_size_stride(arg12_1, (24, ), (1, ))
    assert_size_stride(arg13_1, (24, ), (1, ))
    assert_size_stride(arg14_1, (48, 24), (24, 1))
    assert_size_stride(arg15_1, (24, 24), (24, 1))
    assert_size_stride(arg16_1, (24, 24), (24, 1))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (24, ), (1, ))
    assert_size_stride(arg19_1, (24, ), (1, ))
    assert_size_stride(arg20_1, (96, 24), (24, 1))
    assert_size_stride(arg21_1, (96, ), (1, ))
    assert_size_stride(arg22_1, (24, 96), (96, 1))
    assert_size_stride(arg23_1, (24, ), (1, ))
    assert_size_stride(arg24_1, (24, ), (1, ))
    assert_size_stride(arg25_1, (24, ), (1, ))
    assert_size_stride(arg26_1, (384, 384), (384, 1))
    assert_size_stride(arg27_1, (384, ), (1, ))
    assert_size_stride(arg28_1, (384, ), (1, ))
    assert_size_stride(arg29_1, (384, ), (1, ))
    assert_size_stride(arg30_1, (768, 384), (384, 1))
    assert_size_stride(arg31_1, (384, 384), (384, 1))
    assert_size_stride(arg32_1, (384, 384), (384, 1))
    assert_size_stride(arg33_1, (384, ), (1, ))
    assert_size_stride(arg34_1, (384, ), (1, ))
    assert_size_stride(arg35_1, (384, ), (1, ))
    assert_size_stride(arg36_1, (1536, 384), (384, 1))
    assert_size_stride(arg37_1, (1536, ), (1, ))
    assert_size_stride(arg38_1, (384, 1536), (1536, 1))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (24, ), (1, ))
    assert_size_stride(arg41_1, (24, ), (1, ))
    assert_size_stride(arg42_1, (48, 24), (24, 1))
    assert_size_stride(arg43_1, (24, 24), (24, 1))
    assert_size_stride(arg44_1, (24, 24), (24, 1))
    assert_size_stride(arg45_1, (24, ), (1, ))
    assert_size_stride(arg46_1, (24, ), (1, ))
    assert_size_stride(arg47_1, (24, ), (1, ))
    assert_size_stride(arg48_1, (96, 24), (24, 1))
    assert_size_stride(arg49_1, (96, ), (1, ))
    assert_size_stride(arg50_1, (24, 96), (96, 1))
    assert_size_stride(arg51_1, (24, ), (1, ))
    assert_size_stride(arg52_1, (24, ), (1, ))
    assert_size_stride(arg53_1, (24, ), (1, ))
    assert_size_stride(arg54_1, (384, 384), (384, 1))
    assert_size_stride(arg55_1, (384, ), (1, ))
    assert_size_stride(arg56_1, (384, ), (1, ))
    assert_size_stride(arg57_1, (384, ), (1, ))
    assert_size_stride(arg58_1, (768, 384), (384, 1))
    assert_size_stride(arg59_1, (384, 384), (384, 1))
    assert_size_stride(arg60_1, (384, 384), (384, 1))
    assert_size_stride(arg61_1, (384, ), (1, ))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (384, ), (1, ))
    assert_size_stride(arg64_1, (1536, 384), (384, 1))
    assert_size_stride(arg65_1, (1536, ), (1, ))
    assert_size_stride(arg66_1, (384, 1536), (1536, 1))
    assert_size_stride(arg67_1, (384, ), (1, ))
    assert_size_stride(arg68_1, (24, ), (1, ))
    assert_size_stride(arg69_1, (24, ), (1, ))
    assert_size_stride(arg70_1, (48, 24), (24, 1))
    assert_size_stride(arg71_1, (24, 24), (24, 1))
    assert_size_stride(arg72_1, (24, 24), (24, 1))
    assert_size_stride(arg73_1, (24, ), (1, ))
    assert_size_stride(arg74_1, (24, ), (1, ))
    assert_size_stride(arg75_1, (24, ), (1, ))
    assert_size_stride(arg76_1, (96, 24), (24, 1))
    assert_size_stride(arg77_1, (96, ), (1, ))
    assert_size_stride(arg78_1, (24, 96), (96, 1))
    assert_size_stride(arg79_1, (24, ), (1, ))
    assert_size_stride(arg80_1, (24, ), (1, ))
    assert_size_stride(arg81_1, (24, ), (1, ))
    assert_size_stride(arg82_1, (384, 384), (384, 1))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (384, ), (1, ))
    assert_size_stride(arg85_1, (384, ), (1, ))
    assert_size_stride(arg86_1, (768, 384), (384, 1))
    assert_size_stride(arg87_1, (384, 384), (384, 1))
    assert_size_stride(arg88_1, (384, 384), (384, 1))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (384, ), (1, ))
    assert_size_stride(arg92_1, (1536, 384), (384, 1))
    assert_size_stride(arg93_1, (1536, ), (1, ))
    assert_size_stride(arg94_1, (384, 1536), (1536, 1))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (24, ), (1, ))
    assert_size_stride(arg97_1, (24, ), (1, ))
    assert_size_stride(arg98_1, (48, 24), (24, 1))
    assert_size_stride(arg99_1, (24, 24), (24, 1))
    assert_size_stride(arg100_1, (24, 24), (24, 1))
    assert_size_stride(arg101_1, (24, ), (1, ))
    assert_size_stride(arg102_1, (24, ), (1, ))
    assert_size_stride(arg103_1, (24, ), (1, ))
    assert_size_stride(arg104_1, (96, 24), (24, 1))
    assert_size_stride(arg105_1, (96, ), (1, ))
    assert_size_stride(arg106_1, (24, 96), (96, 1))
    assert_size_stride(arg107_1, (24, ), (1, ))
    assert_size_stride(arg108_1, (24, ), (1, ))
    assert_size_stride(arg109_1, (24, ), (1, ))
    assert_size_stride(arg110_1, (384, 384), (384, 1))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (384, ), (1, ))
    assert_size_stride(arg114_1, (768, 384), (384, 1))
    assert_size_stride(arg115_1, (384, 384), (384, 1))
    assert_size_stride(arg116_1, (384, 384), (384, 1))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (384, ), (1, ))
    assert_size_stride(arg119_1, (384, ), (1, ))
    assert_size_stride(arg120_1, (1536, 384), (384, 1))
    assert_size_stride(arg121_1, (1536, ), (1, ))
    assert_size_stride(arg122_1, (384, 1536), (1536, 1))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (24, ), (1, ))
    assert_size_stride(arg125_1, (24, ), (1, ))
    assert_size_stride(arg126_1, (48, 24), (24, 1))
    assert_size_stride(arg127_1, (24, 24), (24, 1))
    assert_size_stride(arg128_1, (24, 24), (24, 1))
    assert_size_stride(arg129_1, (24, ), (1, ))
    assert_size_stride(arg130_1, (24, ), (1, ))
    assert_size_stride(arg131_1, (24, ), (1, ))
    assert_size_stride(arg132_1, (96, 24), (24, 1))
    assert_size_stride(arg133_1, (96, ), (1, ))
    assert_size_stride(arg134_1, (24, 96), (96, 1))
    assert_size_stride(arg135_1, (24, ), (1, ))
    assert_size_stride(arg136_1, (24, ), (1, ))
    assert_size_stride(arg137_1, (24, ), (1, ))
    assert_size_stride(arg138_1, (384, 384), (384, 1))
    assert_size_stride(arg139_1, (384, ), (1, ))
    assert_size_stride(arg140_1, (384, ), (1, ))
    assert_size_stride(arg141_1, (384, ), (1, ))
    assert_size_stride(arg142_1, (768, 384), (384, 1))
    assert_size_stride(arg143_1, (384, 384), (384, 1))
    assert_size_stride(arg144_1, (384, 384), (384, 1))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (384, ), (1, ))
    assert_size_stride(arg147_1, (384, ), (1, ))
    assert_size_stride(arg148_1, (1536, 384), (384, 1))
    assert_size_stride(arg149_1, (1536, ), (1, ))
    assert_size_stride(arg150_1, (384, 1536), (1536, 1))
    assert_size_stride(arg151_1, (384, ), (1, ))
    assert_size_stride(arg152_1, (24, ), (1, ))
    assert_size_stride(arg153_1, (24, ), (1, ))
    assert_size_stride(arg154_1, (48, 24), (24, 1))
    assert_size_stride(arg155_1, (24, 24), (24, 1))
    assert_size_stride(arg156_1, (24, 24), (24, 1))
    assert_size_stride(arg157_1, (24, ), (1, ))
    assert_size_stride(arg158_1, (24, ), (1, ))
    assert_size_stride(arg159_1, (24, ), (1, ))
    assert_size_stride(arg160_1, (96, 24), (24, 1))
    assert_size_stride(arg161_1, (96, ), (1, ))
    assert_size_stride(arg162_1, (24, 96), (96, 1))
    assert_size_stride(arg163_1, (24, ), (1, ))
    assert_size_stride(arg164_1, (24, ), (1, ))
    assert_size_stride(arg165_1, (24, ), (1, ))
    assert_size_stride(arg166_1, (384, 384), (384, 1))
    assert_size_stride(arg167_1, (384, ), (1, ))
    assert_size_stride(arg168_1, (384, ), (1, ))
    assert_size_stride(arg169_1, (384, ), (1, ))
    assert_size_stride(arg170_1, (768, 384), (384, 1))
    assert_size_stride(arg171_1, (384, 384), (384, 1))
    assert_size_stride(arg172_1, (384, 384), (384, 1))
    assert_size_stride(arg173_1, (384, ), (1, ))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (384, ), (1, ))
    assert_size_stride(arg176_1, (1536, 384), (384, 1))
    assert_size_stride(arg177_1, (1536, ), (1, ))
    assert_size_stride(arg178_1, (384, 1536), (1536, 1))
    assert_size_stride(arg179_1, (384, ), (1, ))
    assert_size_stride(arg180_1, (24, ), (1, ))
    assert_size_stride(arg181_1, (24, ), (1, ))
    assert_size_stride(arg182_1, (48, 24), (24, 1))
    assert_size_stride(arg183_1, (24, 24), (24, 1))
    assert_size_stride(arg184_1, (24, 24), (24, 1))
    assert_size_stride(arg185_1, (24, ), (1, ))
    assert_size_stride(arg186_1, (24, ), (1, ))
    assert_size_stride(arg187_1, (24, ), (1, ))
    assert_size_stride(arg188_1, (96, 24), (24, 1))
    assert_size_stride(arg189_1, (96, ), (1, ))
    assert_size_stride(arg190_1, (24, 96), (96, 1))
    assert_size_stride(arg191_1, (24, ), (1, ))
    assert_size_stride(arg192_1, (24, ), (1, ))
    assert_size_stride(arg193_1, (24, ), (1, ))
    assert_size_stride(arg194_1, (384, 384), (384, 1))
    assert_size_stride(arg195_1, (384, ), (1, ))
    assert_size_stride(arg196_1, (384, ), (1, ))
    assert_size_stride(arg197_1, (384, ), (1, ))
    assert_size_stride(arg198_1, (768, 384), (384, 1))
    assert_size_stride(arg199_1, (384, 384), (384, 1))
    assert_size_stride(arg200_1, (384, 384), (384, 1))
    assert_size_stride(arg201_1, (384, ), (1, ))
    assert_size_stride(arg202_1, (384, ), (1, ))
    assert_size_stride(arg203_1, (384, ), (1, ))
    assert_size_stride(arg204_1, (1536, 384), (384, 1))
    assert_size_stride(arg205_1, (1536, ), (1, ))
    assert_size_stride(arg206_1, (384, 1536), (1536, 1))
    assert_size_stride(arg207_1, (384, ), (1, ))
    assert_size_stride(arg208_1, (24, ), (1, ))
    assert_size_stride(arg209_1, (24, ), (1, ))
    assert_size_stride(arg210_1, (48, 24), (24, 1))
    assert_size_stride(arg211_1, (24, 24), (24, 1))
    assert_size_stride(arg212_1, (24, 24), (24, 1))
    assert_size_stride(arg213_1, (24, ), (1, ))
    assert_size_stride(arg214_1, (24, ), (1, ))
    assert_size_stride(arg215_1, (24, ), (1, ))
    assert_size_stride(arg216_1, (96, 24), (24, 1))
    assert_size_stride(arg217_1, (96, ), (1, ))
    assert_size_stride(arg218_1, (24, 96), (96, 1))
    assert_size_stride(arg219_1, (24, ), (1, ))
    assert_size_stride(arg220_1, (24, ), (1, ))
    assert_size_stride(arg221_1, (24, ), (1, ))
    assert_size_stride(arg222_1, (384, 384), (384, 1))
    assert_size_stride(arg223_1, (384, ), (1, ))
    assert_size_stride(arg224_1, (384, ), (1, ))
    assert_size_stride(arg225_1, (384, ), (1, ))
    assert_size_stride(arg226_1, (768, 384), (384, 1))
    assert_size_stride(arg227_1, (384, 384), (384, 1))
    assert_size_stride(arg228_1, (384, 384), (384, 1))
    assert_size_stride(arg229_1, (384, ), (1, ))
    assert_size_stride(arg230_1, (384, ), (1, ))
    assert_size_stride(arg231_1, (384, ), (1, ))
    assert_size_stride(arg232_1, (1536, 384), (384, 1))
    assert_size_stride(arg233_1, (1536, ), (1, ))
    assert_size_stride(arg234_1, (384, 1536), (1536, 1))
    assert_size_stride(arg235_1, (384, ), (1, ))
    assert_size_stride(arg236_1, (24, ), (1, ))
    assert_size_stride(arg237_1, (24, ), (1, ))
    assert_size_stride(arg238_1, (48, 24), (24, 1))
    assert_size_stride(arg239_1, (24, 24), (24, 1))
    assert_size_stride(arg240_1, (24, 24), (24, 1))
    assert_size_stride(arg241_1, (24, ), (1, ))
    assert_size_stride(arg242_1, (24, ), (1, ))
    assert_size_stride(arg243_1, (24, ), (1, ))
    assert_size_stride(arg244_1, (96, 24), (24, 1))
    assert_size_stride(arg245_1, (96, ), (1, ))
    assert_size_stride(arg246_1, (24, 96), (96, 1))
    assert_size_stride(arg247_1, (24, ), (1, ))
    assert_size_stride(arg248_1, (24, ), (1, ))
    assert_size_stride(arg249_1, (24, ), (1, ))
    assert_size_stride(arg250_1, (384, 384), (384, 1))
    assert_size_stride(arg251_1, (384, ), (1, ))
    assert_size_stride(arg252_1, (384, ), (1, ))
    assert_size_stride(arg253_1, (384, ), (1, ))
    assert_size_stride(arg254_1, (768, 384), (384, 1))
    assert_size_stride(arg255_1, (384, 384), (384, 1))
    assert_size_stride(arg256_1, (384, 384), (384, 1))
    assert_size_stride(arg257_1, (384, ), (1, ))
    assert_size_stride(arg258_1, (384, ), (1, ))
    assert_size_stride(arg259_1, (384, ), (1, ))
    assert_size_stride(arg260_1, (1536, 384), (384, 1))
    assert_size_stride(arg261_1, (1536, ), (1, ))
    assert_size_stride(arg262_1, (384, 1536), (1536, 1))
    assert_size_stride(arg263_1, (384, ), (1, ))
    assert_size_stride(arg264_1, (24, ), (1, ))
    assert_size_stride(arg265_1, (24, ), (1, ))
    assert_size_stride(arg266_1, (48, 24), (24, 1))
    assert_size_stride(arg267_1, (24, 24), (24, 1))
    assert_size_stride(arg268_1, (24, 24), (24, 1))
    assert_size_stride(arg269_1, (24, ), (1, ))
    assert_size_stride(arg270_1, (24, ), (1, ))
    assert_size_stride(arg271_1, (24, ), (1, ))
    assert_size_stride(arg272_1, (96, 24), (24, 1))
    assert_size_stride(arg273_1, (96, ), (1, ))
    assert_size_stride(arg274_1, (24, 96), (96, 1))
    assert_size_stride(arg275_1, (24, ), (1, ))
    assert_size_stride(arg276_1, (24, ), (1, ))
    assert_size_stride(arg277_1, (24, ), (1, ))
    assert_size_stride(arg278_1, (384, 384), (384, 1))
    assert_size_stride(arg279_1, (384, ), (1, ))
    assert_size_stride(arg280_1, (384, ), (1, ))
    assert_size_stride(arg281_1, (384, ), (1, ))
    assert_size_stride(arg282_1, (768, 384), (384, 1))
    assert_size_stride(arg283_1, (384, 384), (384, 1))
    assert_size_stride(arg284_1, (384, 384), (384, 1))
    assert_size_stride(arg285_1, (384, ), (1, ))
    assert_size_stride(arg286_1, (384, ), (1, ))
    assert_size_stride(arg287_1, (384, ), (1, ))
    assert_size_stride(arg288_1, (1536, 384), (384, 1))
    assert_size_stride(arg289_1, (1536, ), (1, ))
    assert_size_stride(arg290_1, (384, 1536), (1536, 1))
    assert_size_stride(arg291_1, (384, ), (1, ))
    assert_size_stride(arg292_1, (24, ), (1, ))
    assert_size_stride(arg293_1, (24, ), (1, ))
    assert_size_stride(arg294_1, (48, 24), (24, 1))
    assert_size_stride(arg295_1, (24, 24), (24, 1))
    assert_size_stride(arg296_1, (24, 24), (24, 1))
    assert_size_stride(arg297_1, (24, ), (1, ))
    assert_size_stride(arg298_1, (24, ), (1, ))
    assert_size_stride(arg299_1, (24, ), (1, ))
    assert_size_stride(arg300_1, (96, 24), (24, 1))
    assert_size_stride(arg301_1, (96, ), (1, ))
    assert_size_stride(arg302_1, (24, 96), (96, 1))
    assert_size_stride(arg303_1, (24, ), (1, ))
    assert_size_stride(arg304_1, (24, ), (1, ))
    assert_size_stride(arg305_1, (24, ), (1, ))
    assert_size_stride(arg306_1, (384, 384), (384, 1))
    assert_size_stride(arg307_1, (384, ), (1, ))
    assert_size_stride(arg308_1, (384, ), (1, ))
    assert_size_stride(arg309_1, (384, ), (1, ))
    assert_size_stride(arg310_1, (768, 384), (384, 1))
    assert_size_stride(arg311_1, (384, 384), (384, 1))
    assert_size_stride(arg312_1, (384, 384), (384, 1))
    assert_size_stride(arg313_1, (384, ), (1, ))
    assert_size_stride(arg314_1, (384, ), (1, ))
    assert_size_stride(arg315_1, (384, ), (1, ))
    assert_size_stride(arg316_1, (1536, 384), (384, 1))
    assert_size_stride(arg317_1, (1536, ), (1, ))
    assert_size_stride(arg318_1, (384, 1536), (1536, 1))
    assert_size_stride(arg319_1, (384, ), (1, ))
    assert_size_stride(arg320_1, (24, ), (1, ))
    assert_size_stride(arg321_1, (24, ), (1, ))
    assert_size_stride(arg322_1, (48, 24), (24, 1))
    assert_size_stride(arg323_1, (24, 24), (24, 1))
    assert_size_stride(arg324_1, (24, 24), (24, 1))
    assert_size_stride(arg325_1, (24, ), (1, ))
    assert_size_stride(arg326_1, (24, ), (1, ))
    assert_size_stride(arg327_1, (24, ), (1, ))
    assert_size_stride(arg328_1, (96, 24), (24, 1))
    assert_size_stride(arg329_1, (96, ), (1, ))
    assert_size_stride(arg330_1, (24, 96), (96, 1))
    assert_size_stride(arg331_1, (24, ), (1, ))
    assert_size_stride(arg332_1, (24, ), (1, ))
    assert_size_stride(arg333_1, (24, ), (1, ))
    assert_size_stride(arg334_1, (384, 384), (384, 1))
    assert_size_stride(arg335_1, (384, ), (1, ))
    assert_size_stride(arg336_1, (384, ), (1, ))
    assert_size_stride(arg337_1, (384, ), (1, ))
    assert_size_stride(arg338_1, (768, 384), (384, 1))
    assert_size_stride(arg339_1, (384, 384), (384, 1))
    assert_size_stride(arg340_1, (384, 384), (384, 1))
    assert_size_stride(arg341_1, (384, ), (1, ))
    assert_size_stride(arg342_1, (384, ), (1, ))
    assert_size_stride(arg343_1, (384, ), (1, ))
    assert_size_stride(arg344_1, (1536, 384), (384, 1))
    assert_size_stride(arg345_1, (1536, ), (1, ))
    assert_size_stride(arg346_1, (384, 1536), (1536, 1))
    assert_size_stride(arg347_1, (384, ), (1, ))
    assert_size_stride(arg348_1, (384, ), (1, ))
    assert_size_stride(arg349_1, (384, ), (1, ))
    assert_size_stride(arg350_1, (1000, 384), (384, 1))
    assert_size_stride(arg351_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_200], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg0_1, arg2_1, stride=(4, 4), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg0_1
        del arg2_1
        buf1 = empty_strided_cuda((8, 196, 1, 3), (588, 3, 4704, 1), torch.float32)
        buf2 = empty_strided_cuda((8, 196, 1, 3), (588, 3, 4704, 1), torch.float32)
        buf3 = empty_strided_cuda((8, 196, 1, 3), (588, 3, 4704, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_63], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf0, arg3_1, arg1_1, buf1, buf2, buf3, 4704, 128, grid=grid(4704), stream=stream0)
        buf4 = empty_strided_cuda((8, 196, 1), (196, 1, 1568), torch.float32)
        buf5 = empty_strided_cuda((8, 196, 1), (196, 1, 1568), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_63], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 1568, 3, grid=grid(1568), stream=stream0)
        del buf1
        del buf2
        del buf3
        buf7 = empty_strided_cuda((8, 196, 384), (75264, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_63], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_2.run(buf0, arg3_1, arg1_1, buf4, buf5, arg4_1, arg5_1, buf7, 602112, grid=grid(602112), stream=stream0)
        del arg4_1
        del arg5_1
        buf8 = empty_strided_cuda((1568, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_134], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg7_1, reinterpret_tensor(buf7, (1568, 384), (384, 1), 0), reinterpret_tensor(arg6_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf8)
        del arg6_1
        del arg7_1
        buf9 = buf5; del buf5  # reuse
        buf10 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_41], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_3.run(buf8, buf9, buf10, 1568, 384, grid=grid(1568), stream=stream0)
        buf12 = empty_strided_cuda((1568, 16, 1), (16, 1, 25088), torch.float32)
        buf13 = empty_strided_cuda((1568, 16, 1), (16, 1, 25088), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_65], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_4.run(buf0, arg3_1, arg1_1, buf12, buf13, 25088, 24, grid=grid(25088), stream=stream0)
        buf15 = reinterpret_tensor(buf7, (1568, 16, 24), (384, 24, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_65], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_5.run(buf0, arg3_1, arg1_1, buf12, buf13, arg12_1, arg13_1, buf15, 37632, 16, grid=grid(37632, 16), stream=stream0)
        del arg12_1
        del arg13_1
        del buf12
        del buf13
        buf16 = empty_strided_cuda((25088, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_135], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (25088, 24), (24, 1), 0), reinterpret_tensor(arg14_1, (24, 48), (1, 24), 0), out=buf16)
        del arg14_1
        buf17 = empty_strided_cuda((1568, 4, 16, 6), (384, 96, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf16, buf17, 602112, grid=grid(602112), stream=stream0)
        buf18 = empty_strided_cuda((1568, 4, 6, 16), (384, 96, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf16, buf18, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf19 = empty_strided_cuda((6272, 16, 16), (256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf17, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf18, (6272, 6, 16), (96, 16, 1), 0), out=buf19)
        buf23 = empty_strided_cuda((1568, 4, 16, 16), (1024, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_73], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf19, buf23, 100352, 16, grid=grid(100352), stream=stream0)
        buf22 = reinterpret_tensor(buf18, (25088, 24), (24, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [linear_136], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (25088, 24), (24, 1), 0), reinterpret_tensor(arg15_1, (24, 24), (1, 24), 0), out=buf22)
        del arg15_1
        buf24 = reinterpret_tensor(buf15, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [matmul_49], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf22, buf24, 602112, grid=grid(602112), stream=stream0)
        buf25 = reinterpret_tensor(buf22, (6272, 16, 6), (96, 6, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [matmul_49], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf23, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf24, (6272, 16, 6), (96, 6, 1), 0), out=buf25)
        buf26 = reinterpret_tensor(buf24, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_205], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf25, buf26, 602112, grid=grid(602112), stream=stream0)
        buf27 = reinterpret_tensor(buf25, (25088, 24), (24, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (25088, 24), (24, 1), 0), reinterpret_tensor(arg16_1, (24, 24), (1, 24), 0), out=buf27)
        del arg16_1
        buf28 = reinterpret_tensor(buf27, (1568, 16, 24), (384, 24, 1), 0); del buf27  # reuse
        buf32 = reinterpret_tensor(buf26, (1568, 16, 24), (384, 24, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_24, layer_norm_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf28, buf0, arg3_1, arg1_1, arg17_1, arg18_1, arg19_1, buf32, 25088, 24, grid=grid(25088), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg1_1
        del arg3_1
        buf33 = empty_strided_cuda((25088, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (25088, 24), (24, 1), 0), reinterpret_tensor(arg20_1, (24, 96), (1, 24), 0), out=buf33)
        del arg20_1
        buf34 = reinterpret_tensor(buf33, (1568, 16, 96), (1536, 96, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_209], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf34, arg21_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg21_1
        buf35 = reinterpret_tensor(buf32, (25088, 24), (24, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (25088, 96), (96, 1), 0), reinterpret_tensor(arg22_1, (96, 24), (1, 96), 0), out=buf35)
        del arg22_1
        buf40 = reinterpret_tensor(buf0, (1568, 16, 24), (384, 24, 1), 0); del buf0  # reuse
        buf66 = reinterpret_tensor(buf17, (1568, 16, 24), (384, 24, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_25, layer_norm_67, layer_norm_70], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf28, buf35, arg23_1, arg24_1, arg25_1, arg40_1, arg41_1, buf40, buf66, 25088, 24, grid=grid(25088), stream=stream0)
        del arg24_1
        del arg25_1
        del arg40_1
        del arg41_1
        buf39 = empty_strided_cuda((8, 197, 384), (75648, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [patch_embed_42, patch_embed_43], Original ATen: [aten.cat, aten.add]
        triton_poi_fused_add_cat_14.run(arg10_1, buf8, buf9, buf10, arg8_1, arg9_1, arg11_1, buf39, 605184, grid=grid(605184), stream=stream0)
        del arg10_1
        del arg11_1
        del arg8_1
        del arg9_1
        del buf10
        del buf9
        buf41 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (1568, 384), (384, 1), 0), reinterpret_tensor(arg26_1, (384, 384), (1, 384), 0), out=buf41)
        del arg26_1
        buf46 = empty_strided_cuda((8, 197, 384), (75648, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [patch_embed_45, layer_norm_68], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(buf39, buf41, arg27_1, arg28_1, arg29_1, buf46, 1576, 384, grid=grid(1576), stream=stream0)
        del arg28_1
        del arg29_1
        buf47 = empty_strided_cuda((1576, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_141], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (1576, 384), (384, 1), 0), reinterpret_tensor(arg30_1, (384, 768), (1, 384), 0), out=buf47)
        del arg30_1
        buf48 = empty_strided_cuda((8, 6, 197, 64), (75648, 12608, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf47, buf48, 605184, grid=grid(605184), stream=stream0)
        buf49 = empty_strided_cuda((8, 6, 64, 197), (75648, 12608, 197, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf47, buf49, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf50 = empty_strided_cuda((48, 197, 197), (38809, 197, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_50], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf49, (48, 64, 197), (12608, 197, 1), 0), out=buf50)
        buf54 = empty_strided_cuda((8, 6, 197, 197), (232896, 38816, 197, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_76], Original ATen: [aten._softmax]
        triton_red_fused__softmax_18.run(buf50, buf54, 9456, 197, grid=grid(9456), stream=stream0)
        buf53 = reinterpret_tensor(buf49, (1576, 384), (384, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [linear_142], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (1576, 384), (384, 1), 0), reinterpret_tensor(arg31_1, (384, 384), (1, 384), 0), out=buf53)
        del arg31_1
        buf55 = reinterpret_tensor(buf46, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [matmul_51], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf53, buf55, 605184, grid=grid(605184), stream=stream0)
        buf56 = reinterpret_tensor(buf53, (48, 197, 64), (12608, 64, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [matmul_51], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (48, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf55, (48, 197, 64), (12608, 64, 1), 0), out=buf56)
        buf57 = reinterpret_tensor(buf55, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_213], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf56, buf57, 605184, grid=grid(605184), stream=stream0)
        buf58 = reinterpret_tensor(buf56, (1576, 384), (384, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf57, (1576, 384), (384, 1), 0), reinterpret_tensor(arg32_1, (384, 384), (1, 384), 0), out=buf58)
        del arg32_1
        buf59 = reinterpret_tensor(buf58, (8, 197, 384), (75648, 384, 1), 0); del buf58  # reuse
        buf90 = reinterpret_tensor(buf57, (8, 197, 384), (75648, 384, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_45, patch_embed_46, layer_norm_69], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_21.run(buf59, buf39, buf41, arg27_1, arg33_1, arg34_1, arg35_1, buf90, 1576, 384, grid=grid(1576), stream=stream0)
        del arg27_1
        del arg33_1
        del arg34_1
        del arg35_1
        buf67 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [linear_146], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (25088, 24), (24, 1), 0), reinterpret_tensor(arg42_1, (24, 48), (1, 24), 0), out=buf67)
        del arg42_1
        buf68 = reinterpret_tensor(buf41, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [matmul_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf67, buf68, 602112, grid=grid(602112), stream=stream0)
        buf69 = reinterpret_tensor(buf40, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [matmul_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf67, buf69, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf70 = reinterpret_tensor(buf23, (6272, 16, 16), (256, 16, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [matmul_52], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf69, (6272, 6, 16), (96, 16, 1), 0), out=buf70)
        buf74 = reinterpret_tensor(buf19, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [attn_79], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf70, buf74, 100352, 16, grid=grid(100352), stream=stream0)
        buf73 = reinterpret_tensor(buf69, (25088, 24), (24, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [linear_147], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (25088, 24), (24, 1), 0), reinterpret_tensor(arg43_1, (24, 24), (1, 24), 0), out=buf73)
        del arg43_1
        buf75 = reinterpret_tensor(buf66, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [matmul_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf73, buf75, 602112, grid=grid(602112), stream=stream0)
        buf76 = reinterpret_tensor(buf73, (6272, 16, 6), (96, 6, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [matmul_53], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf74, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf75, (6272, 16, 6), (96, 6, 1), 0), out=buf76)
        buf77 = reinterpret_tensor(buf75, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf76, buf77, 602112, grid=grid(602112), stream=stream0)
        buf78 = reinterpret_tensor(buf76, (25088, 24), (24, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (25088, 24), (24, 1), 0), reinterpret_tensor(arg44_1, (24, 24), (1, 24), 0), out=buf78)
        del arg44_1
        buf79 = reinterpret_tensor(buf78, (1568, 16, 24), (384, 24, 1), 0); del buf78  # reuse
        buf83 = reinterpret_tensor(buf77, (1568, 16, 24), (384, 24, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_25, pixel_embed_26, layer_norm_71], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf79, buf28, buf35, arg23_1, arg45_1, arg46_1, arg47_1, buf83, 25088, 24, grid=grid(25088), stream=stream0)
        del arg23_1
        del arg45_1
        del arg46_1
        del arg47_1
        buf84 = reinterpret_tensor(buf34, (25088, 96), (96, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf83, (25088, 24), (24, 1), 0), reinterpret_tensor(arg48_1, (24, 96), (1, 24), 0), out=buf84)
        del arg48_1
        buf85 = reinterpret_tensor(buf84, (1568, 16, 96), (1536, 96, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_225], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf85, arg49_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg49_1
        buf86 = reinterpret_tensor(buf83, (25088, 24), (24, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf85, (25088, 96), (96, 1), 0), reinterpret_tensor(arg50_1, (96, 24), (1, 96), 0), out=buf86)
        del arg50_1
        buf94 = reinterpret_tensor(buf35, (1568, 16, 24), (384, 24, 1), 0); del buf35  # reuse
        buf119 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_27, layer_norm_72, layer_norm_75], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf79, buf86, arg51_1, arg52_1, arg53_1, arg68_1, arg69_1, buf94, buf119, 25088, 24, grid=grid(25088), stream=stream0)
        del arg52_1
        del arg53_1
        del arg68_1
        del arg69_1
        buf91 = empty_strided_cuda((1576, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (1576, 384), (384, 1), 0), reinterpret_tensor(arg36_1, (384, 1536), (1, 384), 0), out=buf91)
        del arg36_1
        buf92 = reinterpret_tensor(buf91, (8, 197, 1536), (302592, 1536, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_217], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf92, arg37_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg37_1
        buf93 = reinterpret_tensor(buf90, (1576, 384), (384, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg38_1, (1536, 384), (1, 1536), 0), out=buf93)
        del arg38_1
        buf95 = reinterpret_tensor(buf68, (1568, 384), (384, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (1568, 384), (384, 1), 0), reinterpret_tensor(arg54_1, (384, 384), (1, 384), 0), out=buf95)
        del arg54_1
        buf96 = buf39; del buf39  # reuse
        buf100 = reinterpret_tensor(buf48, (8, 197, 384), (75648, 384, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_48, layer_norm_73], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_24.run(buf59, buf93, arg39_1, buf95, arg55_1, arg56_1, arg57_1, buf96, buf100, 1576, 384, grid=grid(1576), stream=stream0)
        del arg39_1
        del arg55_1
        del arg56_1
        del arg57_1
        buf101 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [linear_152], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (1576, 384), (384, 1), 0), reinterpret_tensor(arg58_1, (384, 768), (1, 384), 0), out=buf101)
        del arg58_1
        buf102 = reinterpret_tensor(buf93, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [matmul_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf101, buf102, 605184, grid=grid(605184), stream=stream0)
        buf103 = reinterpret_tensor(buf59, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [matmul_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf101, buf103, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf104 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [matmul_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf102, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf103, (48, 64, 197), (12608, 197, 1), 0), out=buf104)
        buf108 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [attn_82], Original ATen: [aten._softmax]
        triton_red_fused__softmax_18.run(buf104, buf108, 9456, 197, grid=grid(9456), stream=stream0)
        buf107 = reinterpret_tensor(buf103, (1576, 384), (384, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [linear_153], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (1576, 384), (384, 1), 0), reinterpret_tensor(arg59_1, (384, 384), (1, 384), 0), out=buf107)
        del arg59_1
        buf109 = reinterpret_tensor(buf100, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [matmul_55], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf107, buf109, 605184, grid=grid(605184), stream=stream0)
        buf110 = reinterpret_tensor(buf107, (48, 197, 64), (12608, 64, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [matmul_55], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf108, (48, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf109, (48, 197, 64), (12608, 64, 1), 0), out=buf110)
        buf111 = reinterpret_tensor(buf109, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_229], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf110, buf111, 605184, grid=grid(605184), stream=stream0)
        buf112 = reinterpret_tensor(buf110, (1576, 384), (384, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (1576, 384), (384, 1), 0), reinterpret_tensor(arg60_1, (384, 384), (1, 384), 0), out=buf112)
        del arg60_1
        buf143 = reinterpret_tensor(buf111, (8, 197, 384), (75648, 384, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_49, layer_norm_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf96, buf112, arg61_1, arg62_1, arg63_1, buf143, 1576, 384, grid=grid(1576), stream=stream0)
        del arg62_1
        del arg63_1
        buf120 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [linear_157], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (25088, 24), (24, 1), 0), reinterpret_tensor(arg70_1, (24, 48), (1, 24), 0), out=buf120)
        del arg70_1
        buf121 = reinterpret_tensor(buf95, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [matmul_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf120, buf121, 602112, grid=grid(602112), stream=stream0)
        buf122 = reinterpret_tensor(buf94, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [matmul_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf120, buf122, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf123 = reinterpret_tensor(buf74, (6272, 16, 16), (256, 16, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [matmul_56], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf122, (6272, 6, 16), (96, 16, 1), 0), out=buf123)
        buf127 = reinterpret_tensor(buf70, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [attn_85], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf123, buf127, 100352, 16, grid=grid(100352), stream=stream0)
        buf126 = reinterpret_tensor(buf122, (25088, 24), (24, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [linear_158], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (25088, 24), (24, 1), 0), reinterpret_tensor(arg71_1, (24, 24), (1, 24), 0), out=buf126)
        del arg71_1
        buf128 = reinterpret_tensor(buf119, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [matmul_57], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf126, buf128, 602112, grid=grid(602112), stream=stream0)
        buf129 = reinterpret_tensor(buf126, (6272, 16, 6), (96, 6, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [matmul_57], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf127, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf128, (6272, 16, 6), (96, 6, 1), 0), out=buf129)
        buf130 = reinterpret_tensor(buf128, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_237], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf129, buf130, 602112, grid=grid(602112), stream=stream0)
        buf131 = reinterpret_tensor(buf129, (25088, 24), (24, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (25088, 24), (24, 1), 0), reinterpret_tensor(arg72_1, (24, 24), (1, 24), 0), out=buf131)
        del arg72_1
        buf132 = reinterpret_tensor(buf131, (1568, 16, 24), (384, 24, 1), 0); del buf131  # reuse
        buf136 = reinterpret_tensor(buf130, (1568, 16, 24), (384, 24, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_27, pixel_embed_28, layer_norm_76], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf132, buf79, buf86, arg51_1, arg73_1, arg74_1, arg75_1, buf136, 25088, 24, grid=grid(25088), stream=stream0)
        del arg51_1
        del arg73_1
        del arg74_1
        del arg75_1
        buf137 = reinterpret_tensor(buf85, (25088, 96), (96, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (25088, 24), (24, 1), 0), reinterpret_tensor(arg76_1, (24, 96), (1, 24), 0), out=buf137)
        del arg76_1
        buf138 = reinterpret_tensor(buf137, (1568, 16, 96), (1536, 96, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [x_241], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf138, arg77_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg77_1
        buf139 = reinterpret_tensor(buf136, (25088, 24), (24, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (25088, 96), (96, 1), 0), reinterpret_tensor(arg78_1, (96, 24), (1, 96), 0), out=buf139)
        del arg78_1
        buf148 = reinterpret_tensor(buf86, (1568, 16, 24), (384, 24, 1), 0); del buf86  # reuse
        buf174 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_29, layer_norm_77, layer_norm_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf132, buf139, arg79_1, arg80_1, arg81_1, arg96_1, arg97_1, buf148, buf174, 25088, 24, grid=grid(25088), stream=stream0)
        del arg80_1
        del arg81_1
        del arg96_1
        del arg97_1
        buf144 = reinterpret_tensor(buf92, (1576, 1536), (1536, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (1576, 384), (384, 1), 0), reinterpret_tensor(arg64_1, (384, 1536), (1, 384), 0), out=buf144)
        del arg64_1
        buf145 = reinterpret_tensor(buf144, (8, 197, 1536), (302592, 1536, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_233], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf145, arg65_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg65_1
        buf146 = reinterpret_tensor(buf143, (1576, 384), (384, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg66_1, (1536, 384), (1, 1536), 0), out=buf146)
        del arg66_1
        buf147 = reinterpret_tensor(buf146, (8, 197, 384), (75648, 384, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_49, patch_embed_50], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf147, buf96, buf112, arg61_1, arg67_1, 605184, grid=grid(605184), stream=stream0)
        del arg61_1
        del arg67_1
        buf149 = reinterpret_tensor(buf121, (1568, 384), (384, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (1568, 384), (384, 1), 0), reinterpret_tensor(arg82_1, (384, 384), (1, 384), 0), out=buf149)
        del arg82_1
        buf154 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_51, layer_norm_78], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(buf147, buf149, arg83_1, arg84_1, arg85_1, buf154, 1576, 384, grid=grid(1576), stream=stream0)
        del arg84_1
        del arg85_1
        buf155 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [linear_163], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (1576, 384), (384, 1), 0), reinterpret_tensor(arg86_1, (384, 768), (1, 384), 0), out=buf155)
        del arg86_1
        buf156 = reinterpret_tensor(buf112, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [matmul_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf155, buf156, 605184, grid=grid(605184), stream=stream0)
        buf157 = reinterpret_tensor(buf102, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [matmul_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf155, buf157, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf158 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [matmul_58], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf156, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf157, (48, 64, 197), (12608, 197, 1), 0), out=buf158)
        buf162 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [attn_88], Original ATen: [aten._softmax]
        triton_red_fused__softmax_18.run(buf158, buf162, 9456, 197, grid=grid(9456), stream=stream0)
        buf161 = reinterpret_tensor(buf157, (1576, 384), (384, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [linear_164], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (1576, 384), (384, 1), 0), reinterpret_tensor(arg87_1, (384, 384), (1, 384), 0), out=buf161)
        del arg87_1
        buf163 = reinterpret_tensor(buf154, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [matmul_59], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf161, buf163, 605184, grid=grid(605184), stream=stream0)
        buf164 = reinterpret_tensor(buf161, (48, 197, 64), (12608, 64, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [matmul_59], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf162, (48, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf163, (48, 197, 64), (12608, 64, 1), 0), out=buf164)
        buf165 = reinterpret_tensor(buf163, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_245], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf164, buf165, 605184, grid=grid(605184), stream=stream0)
        buf166 = reinterpret_tensor(buf164, (1576, 384), (384, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (1576, 384), (384, 1), 0), reinterpret_tensor(arg88_1, (384, 384), (1, 384), 0), out=buf166)
        del arg88_1
        buf167 = reinterpret_tensor(buf166, (8, 197, 384), (75648, 384, 1), 0); del buf166  # reuse
        buf198 = reinterpret_tensor(buf165, (8, 197, 384), (75648, 384, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_51, patch_embed_52, layer_norm_79], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_21.run(buf167, buf147, buf149, arg83_1, arg89_1, arg90_1, arg91_1, buf198, 1576, 384, grid=grid(1576), stream=stream0)
        del arg83_1
        del arg89_1
        del arg90_1
        del arg91_1
        buf175 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [linear_168], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf174, (25088, 24), (24, 1), 0), reinterpret_tensor(arg98_1, (24, 48), (1, 24), 0), out=buf175)
        del arg98_1
        buf176 = reinterpret_tensor(buf149, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [matmul_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf175, buf176, 602112, grid=grid(602112), stream=stream0)
        buf177 = reinterpret_tensor(buf148, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [matmul_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf175, buf177, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf178 = reinterpret_tensor(buf127, (6272, 16, 16), (256, 16, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [matmul_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf176, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf177, (6272, 6, 16), (96, 16, 1), 0), out=buf178)
        buf182 = reinterpret_tensor(buf123, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [attn_91], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf178, buf182, 100352, 16, grid=grid(100352), stream=stream0)
        buf181 = reinterpret_tensor(buf177, (25088, 24), (24, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [linear_169], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf174, (25088, 24), (24, 1), 0), reinterpret_tensor(arg99_1, (24, 24), (1, 24), 0), out=buf181)
        del arg99_1
        buf183 = reinterpret_tensor(buf174, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [matmul_61], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf181, buf183, 602112, grid=grid(602112), stream=stream0)
        buf184 = reinterpret_tensor(buf181, (6272, 16, 6), (96, 6, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [matmul_61], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf182, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf183, (6272, 16, 6), (96, 6, 1), 0), out=buf184)
        buf185 = reinterpret_tensor(buf183, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [x_253], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf184, buf185, 602112, grid=grid(602112), stream=stream0)
        buf186 = reinterpret_tensor(buf184, (25088, 24), (24, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf185, (25088, 24), (24, 1), 0), reinterpret_tensor(arg100_1, (24, 24), (1, 24), 0), out=buf186)
        del arg100_1
        buf187 = reinterpret_tensor(buf186, (1568, 16, 24), (384, 24, 1), 0); del buf186  # reuse
        buf191 = reinterpret_tensor(buf185, (1568, 16, 24), (384, 24, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_29, pixel_embed_30, layer_norm_81], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf187, buf132, buf139, arg79_1, arg101_1, arg102_1, arg103_1, buf191, 25088, 24, grid=grid(25088), stream=stream0)
        del arg101_1
        del arg102_1
        del arg103_1
        del arg79_1
        buf192 = reinterpret_tensor(buf138, (25088, 96), (96, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (25088, 24), (24, 1), 0), reinterpret_tensor(arg104_1, (24, 96), (1, 24), 0), out=buf192)
        del arg104_1
        buf193 = reinterpret_tensor(buf192, (1568, 16, 96), (1536, 96, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [x_257], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf193, arg105_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg105_1
        buf194 = reinterpret_tensor(buf191, (25088, 24), (24, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf193, (25088, 96), (96, 1), 0), reinterpret_tensor(arg106_1, (96, 24), (1, 96), 0), out=buf194)
        del arg106_1
        buf202 = reinterpret_tensor(buf139, (1568, 16, 24), (384, 24, 1), 0); del buf139  # reuse
        buf227 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_31, layer_norm_82, layer_norm_85], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf187, buf194, arg107_1, arg108_1, arg109_1, arg124_1, arg125_1, buf202, buf227, 25088, 24, grid=grid(25088), stream=stream0)
        del arg108_1
        del arg109_1
        del arg124_1
        del arg125_1
        buf199 = reinterpret_tensor(buf145, (1576, 1536), (1536, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (1576, 384), (384, 1), 0), reinterpret_tensor(arg92_1, (384, 1536), (1, 384), 0), out=buf199)
        del arg92_1
        buf200 = reinterpret_tensor(buf199, (8, 197, 1536), (302592, 1536, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [x_249], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf200, arg93_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg93_1
        buf201 = reinterpret_tensor(buf198, (1576, 384), (384, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg94_1, (1536, 384), (1, 1536), 0), out=buf201)
        del arg94_1
        buf203 = reinterpret_tensor(buf176, (1568, 384), (384, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf202, (1568, 384), (384, 1), 0), reinterpret_tensor(arg110_1, (384, 384), (1, 384), 0), out=buf203)
        del arg110_1
        buf204 = buf147; del buf147  # reuse
        buf208 = reinterpret_tensor(buf156, (8, 197, 384), (75648, 384, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_54, layer_norm_83], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_24.run(buf167, buf201, arg95_1, buf203, arg111_1, arg112_1, arg113_1, buf204, buf208, 1576, 384, grid=grid(1576), stream=stream0)
        del arg111_1
        del arg112_1
        del arg113_1
        del arg95_1
        buf209 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [linear_174], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (1576, 384), (384, 1), 0), reinterpret_tensor(arg114_1, (384, 768), (1, 384), 0), out=buf209)
        del arg114_1
        buf210 = reinterpret_tensor(buf201, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [matmul_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf209, buf210, 605184, grid=grid(605184), stream=stream0)
        buf211 = reinterpret_tensor(buf167, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [matmul_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf209, buf211, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf212 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [matmul_62], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf210, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf211, (48, 64, 197), (12608, 197, 1), 0), out=buf212)
        buf216 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [attn_94], Original ATen: [aten._softmax]
        triton_red_fused__softmax_18.run(buf212, buf216, 9456, 197, grid=grid(9456), stream=stream0)
        buf215 = reinterpret_tensor(buf211, (1576, 384), (384, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [linear_175], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (1576, 384), (384, 1), 0), reinterpret_tensor(arg115_1, (384, 384), (1, 384), 0), out=buf215)
        del arg115_1
        buf217 = reinterpret_tensor(buf208, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [matmul_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf215, buf217, 605184, grid=grid(605184), stream=stream0)
        buf218 = reinterpret_tensor(buf215, (48, 197, 64), (12608, 64, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [matmul_63], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (48, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf217, (48, 197, 64), (12608, 64, 1), 0), out=buf218)
        buf219 = reinterpret_tensor(buf217, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_261], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf218, buf219, 605184, grid=grid(605184), stream=stream0)
        buf220 = reinterpret_tensor(buf218, (1576, 384), (384, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (1576, 384), (384, 1), 0), reinterpret_tensor(arg116_1, (384, 384), (1, 384), 0), out=buf220)
        del arg116_1
        buf251 = reinterpret_tensor(buf219, (8, 197, 384), (75648, 384, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_55, layer_norm_84], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf204, buf220, arg117_1, arg118_1, arg119_1, buf251, 1576, 384, grid=grid(1576), stream=stream0)
        del arg118_1
        del arg119_1
        buf228 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [linear_179], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (25088, 24), (24, 1), 0), reinterpret_tensor(arg126_1, (24, 48), (1, 24), 0), out=buf228)
        del arg126_1
        buf229 = reinterpret_tensor(buf203, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [matmul_64], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf228, buf229, 602112, grid=grid(602112), stream=stream0)
        buf230 = reinterpret_tensor(buf202, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [matmul_64], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf228, buf230, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf231 = reinterpret_tensor(buf182, (6272, 16, 16), (256, 16, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [matmul_64], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf229, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf230, (6272, 6, 16), (96, 16, 1), 0), out=buf231)
        buf235 = reinterpret_tensor(buf178, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [attn_97], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf231, buf235, 100352, 16, grid=grid(100352), stream=stream0)
        buf234 = reinterpret_tensor(buf230, (25088, 24), (24, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [linear_180], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (25088, 24), (24, 1), 0), reinterpret_tensor(arg127_1, (24, 24), (1, 24), 0), out=buf234)
        del arg127_1
        buf236 = reinterpret_tensor(buf227, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [matmul_65], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf234, buf236, 602112, grid=grid(602112), stream=stream0)
        buf237 = reinterpret_tensor(buf234, (6272, 16, 6), (96, 6, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [matmul_65], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf235, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf236, (6272, 16, 6), (96, 6, 1), 0), out=buf237)
        buf238 = reinterpret_tensor(buf236, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [x_269], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf237, buf238, 602112, grid=grid(602112), stream=stream0)
        buf239 = reinterpret_tensor(buf237, (25088, 24), (24, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (25088, 24), (24, 1), 0), reinterpret_tensor(arg128_1, (24, 24), (1, 24), 0), out=buf239)
        del arg128_1
        buf240 = reinterpret_tensor(buf239, (1568, 16, 24), (384, 24, 1), 0); del buf239  # reuse
        buf244 = reinterpret_tensor(buf238, (1568, 16, 24), (384, 24, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_31, pixel_embed_32, layer_norm_86], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf240, buf187, buf194, arg107_1, arg129_1, arg130_1, arg131_1, buf244, 25088, 24, grid=grid(25088), stream=stream0)
        del arg107_1
        del arg129_1
        del arg130_1
        del arg131_1
        buf245 = reinterpret_tensor(buf193, (25088, 96), (96, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (25088, 24), (24, 1), 0), reinterpret_tensor(arg132_1, (24, 96), (1, 24), 0), out=buf245)
        del arg132_1
        buf246 = reinterpret_tensor(buf245, (1568, 16, 96), (1536, 96, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [x_273], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf246, arg133_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg133_1
        buf247 = reinterpret_tensor(buf244, (25088, 24), (24, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf246, (25088, 96), (96, 1), 0), reinterpret_tensor(arg134_1, (96, 24), (1, 96), 0), out=buf247)
        del arg134_1
        buf256 = reinterpret_tensor(buf194, (1568, 16, 24), (384, 24, 1), 0); del buf194  # reuse
        buf282 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_33, layer_norm_87, layer_norm_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf240, buf247, arg135_1, arg136_1, arg137_1, arg152_1, arg153_1, buf256, buf282, 25088, 24, grid=grid(25088), stream=stream0)
        del arg136_1
        del arg137_1
        del arg152_1
        del arg153_1
        buf252 = reinterpret_tensor(buf200, (1576, 1536), (1536, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf251, (1576, 384), (384, 1), 0), reinterpret_tensor(arg120_1, (384, 1536), (1, 384), 0), out=buf252)
        del arg120_1
        buf253 = reinterpret_tensor(buf252, (8, 197, 1536), (302592, 1536, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [x_265], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf253, arg121_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg121_1
        buf254 = reinterpret_tensor(buf251, (1576, 384), (384, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg122_1, (1536, 384), (1, 1536), 0), out=buf254)
        del arg122_1
        buf255 = reinterpret_tensor(buf254, (8, 197, 384), (75648, 384, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_55, patch_embed_56], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf255, buf204, buf220, arg117_1, arg123_1, 605184, grid=grid(605184), stream=stream0)
        del arg117_1
        del arg123_1
        buf257 = reinterpret_tensor(buf229, (1568, 384), (384, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (1568, 384), (384, 1), 0), reinterpret_tensor(arg138_1, (384, 384), (1, 384), 0), out=buf257)
        del arg138_1
        buf262 = reinterpret_tensor(buf220, (8, 197, 384), (75648, 384, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_57, layer_norm_88], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(buf255, buf257, arg139_1, arg140_1, arg141_1, buf262, 1576, 384, grid=grid(1576), stream=stream0)
        del arg140_1
        del arg141_1
        buf263 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [linear_185], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (1576, 384), (384, 1), 0), reinterpret_tensor(arg142_1, (384, 768), (1, 384), 0), out=buf263)
        del arg142_1
        buf264 = reinterpret_tensor(buf204, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [matmul_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf263, buf264, 605184, grid=grid(605184), stream=stream0)
        buf265 = reinterpret_tensor(buf210, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [matmul_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf263, buf265, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf266 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [matmul_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf264, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf265, (48, 64, 197), (12608, 197, 1), 0), out=buf266)
        buf270 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [attn_100], Original ATen: [aten._softmax]
        triton_red_fused__softmax_18.run(buf266, buf270, 9456, 197, grid=grid(9456), stream=stream0)
        buf269 = reinterpret_tensor(buf265, (1576, 384), (384, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [linear_186], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (1576, 384), (384, 1), 0), reinterpret_tensor(arg143_1, (384, 384), (1, 384), 0), out=buf269)
        del arg143_1
        buf271 = reinterpret_tensor(buf262, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [matmul_67], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf269, buf271, 605184, grid=grid(605184), stream=stream0)
        buf272 = reinterpret_tensor(buf269, (48, 197, 64), (12608, 64, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [matmul_67], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf270, (48, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf271, (48, 197, 64), (12608, 64, 1), 0), out=buf272)
        buf273 = reinterpret_tensor(buf271, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [x_277], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf272, buf273, 605184, grid=grid(605184), stream=stream0)
        buf274 = reinterpret_tensor(buf272, (1576, 384), (384, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf273, (1576, 384), (384, 1), 0), reinterpret_tensor(arg144_1, (384, 384), (1, 384), 0), out=buf274)
        del arg144_1
        buf275 = reinterpret_tensor(buf274, (8, 197, 384), (75648, 384, 1), 0); del buf274  # reuse
        buf306 = reinterpret_tensor(buf273, (8, 197, 384), (75648, 384, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_57, patch_embed_58, layer_norm_89], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_21.run(buf275, buf255, buf257, arg139_1, arg145_1, arg146_1, arg147_1, buf306, 1576, 384, grid=grid(1576), stream=stream0)
        del arg139_1
        del arg145_1
        del arg146_1
        del arg147_1
        buf283 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [linear_190], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (25088, 24), (24, 1), 0), reinterpret_tensor(arg154_1, (24, 48), (1, 24), 0), out=buf283)
        del arg154_1
        buf284 = reinterpret_tensor(buf257, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [matmul_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf283, buf284, 602112, grid=grid(602112), stream=stream0)
        buf285 = reinterpret_tensor(buf256, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [matmul_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf283, buf285, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf286 = reinterpret_tensor(buf235, (6272, 16, 16), (256, 16, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [matmul_68], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf284, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf285, (6272, 6, 16), (96, 16, 1), 0), out=buf286)
        buf290 = reinterpret_tensor(buf231, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [attn_103], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf286, buf290, 100352, 16, grid=grid(100352), stream=stream0)
        buf289 = reinterpret_tensor(buf285, (25088, 24), (24, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [linear_191], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (25088, 24), (24, 1), 0), reinterpret_tensor(arg155_1, (24, 24), (1, 24), 0), out=buf289)
        del arg155_1
        buf291 = reinterpret_tensor(buf282, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [matmul_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf289, buf291, 602112, grid=grid(602112), stream=stream0)
        buf292 = reinterpret_tensor(buf289, (6272, 16, 6), (96, 6, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [matmul_69], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf290, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf291, (6272, 16, 6), (96, 6, 1), 0), out=buf292)
        buf293 = reinterpret_tensor(buf291, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [x_285], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf292, buf293, 602112, grid=grid(602112), stream=stream0)
        buf294 = reinterpret_tensor(buf292, (25088, 24), (24, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf293, (25088, 24), (24, 1), 0), reinterpret_tensor(arg156_1, (24, 24), (1, 24), 0), out=buf294)
        del arg156_1
        buf295 = reinterpret_tensor(buf294, (1568, 16, 24), (384, 24, 1), 0); del buf294  # reuse
        buf299 = reinterpret_tensor(buf293, (1568, 16, 24), (384, 24, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_33, pixel_embed_34, layer_norm_91], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf295, buf240, buf247, arg135_1, arg157_1, arg158_1, arg159_1, buf299, 25088, 24, grid=grid(25088), stream=stream0)
        del arg135_1
        del arg157_1
        del arg158_1
        del arg159_1
        buf300 = reinterpret_tensor(buf246, (25088, 96), (96, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf299, (25088, 24), (24, 1), 0), reinterpret_tensor(arg160_1, (24, 96), (1, 24), 0), out=buf300)
        del arg160_1
        buf301 = reinterpret_tensor(buf300, (1568, 16, 96), (1536, 96, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [x_289], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf301, arg161_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg161_1
        buf302 = reinterpret_tensor(buf299, (25088, 24), (24, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf301, (25088, 96), (96, 1), 0), reinterpret_tensor(arg162_1, (96, 24), (1, 96), 0), out=buf302)
        del arg162_1
        buf310 = reinterpret_tensor(buf247, (1568, 16, 24), (384, 24, 1), 0); del buf247  # reuse
        buf335 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_35, layer_norm_92, layer_norm_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf295, buf302, arg163_1, arg164_1, arg165_1, arg180_1, arg181_1, buf310, buf335, 25088, 24, grid=grid(25088), stream=stream0)
        del arg164_1
        del arg165_1
        del arg180_1
        del arg181_1
        buf307 = reinterpret_tensor(buf253, (1576, 1536), (1536, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf306, (1576, 384), (384, 1), 0), reinterpret_tensor(arg148_1, (384, 1536), (1, 384), 0), out=buf307)
        del arg148_1
        buf308 = reinterpret_tensor(buf307, (8, 197, 1536), (302592, 1536, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [x_281], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf308, arg149_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg149_1
        buf309 = reinterpret_tensor(buf306, (1576, 384), (384, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf308, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg150_1, (1536, 384), (1, 1536), 0), out=buf309)
        del arg150_1
        buf311 = reinterpret_tensor(buf284, (1568, 384), (384, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (1568, 384), (384, 1), 0), reinterpret_tensor(arg166_1, (384, 384), (1, 384), 0), out=buf311)
        del arg166_1
        buf312 = buf255; del buf255  # reuse
        buf316 = reinterpret_tensor(buf264, (8, 197, 384), (75648, 384, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_60, layer_norm_93], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_24.run(buf275, buf309, arg151_1, buf311, arg167_1, arg168_1, arg169_1, buf312, buf316, 1576, 384, grid=grid(1576), stream=stream0)
        del arg151_1
        del arg167_1
        del arg168_1
        del arg169_1
        buf317 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [linear_196], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (1576, 384), (384, 1), 0), reinterpret_tensor(arg170_1, (384, 768), (1, 384), 0), out=buf317)
        del arg170_1
        buf318 = reinterpret_tensor(buf309, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [matmul_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf317, buf318, 605184, grid=grid(605184), stream=stream0)
        buf319 = reinterpret_tensor(buf275, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [matmul_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf317, buf319, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf320 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [matmul_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf318, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf319, (48, 64, 197), (12608, 197, 1), 0), out=buf320)
        buf324 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [attn_106], Original ATen: [aten._softmax]
        triton_red_fused__softmax_18.run(buf320, buf324, 9456, 197, grid=grid(9456), stream=stream0)
        buf323 = reinterpret_tensor(buf319, (1576, 384), (384, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [linear_197], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (1576, 384), (384, 1), 0), reinterpret_tensor(arg171_1, (384, 384), (1, 384), 0), out=buf323)
        del arg171_1
        buf325 = reinterpret_tensor(buf316, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [matmul_71], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf323, buf325, 605184, grid=grid(605184), stream=stream0)
        buf326 = reinterpret_tensor(buf323, (48, 197, 64), (12608, 64, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [matmul_71], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf324, (48, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf325, (48, 197, 64), (12608, 64, 1), 0), out=buf326)
        buf327 = reinterpret_tensor(buf325, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [x_293], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf326, buf327, 605184, grid=grid(605184), stream=stream0)
        buf328 = reinterpret_tensor(buf326, (1576, 384), (384, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf327, (1576, 384), (384, 1), 0), reinterpret_tensor(arg172_1, (384, 384), (1, 384), 0), out=buf328)
        del arg172_1
        buf359 = reinterpret_tensor(buf327, (8, 197, 384), (75648, 384, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_61, layer_norm_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf312, buf328, arg173_1, arg174_1, arg175_1, buf359, 1576, 384, grid=grid(1576), stream=stream0)
        del arg174_1
        del arg175_1
        buf336 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [linear_201], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (25088, 24), (24, 1), 0), reinterpret_tensor(arg182_1, (24, 48), (1, 24), 0), out=buf336)
        del arg182_1
        buf337 = reinterpret_tensor(buf311, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [matmul_72], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf336, buf337, 602112, grid=grid(602112), stream=stream0)
        buf338 = reinterpret_tensor(buf310, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [matmul_72], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf336, buf338, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf339 = reinterpret_tensor(buf290, (6272, 16, 16), (256, 16, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [matmul_72], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf337, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf338, (6272, 6, 16), (96, 16, 1), 0), out=buf339)
        buf343 = reinterpret_tensor(buf286, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [attn_109], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf339, buf343, 100352, 16, grid=grid(100352), stream=stream0)
        buf342 = reinterpret_tensor(buf338, (25088, 24), (24, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [linear_202], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (25088, 24), (24, 1), 0), reinterpret_tensor(arg183_1, (24, 24), (1, 24), 0), out=buf342)
        del arg183_1
        buf344 = reinterpret_tensor(buf335, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [matmul_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf342, buf344, 602112, grid=grid(602112), stream=stream0)
        buf345 = reinterpret_tensor(buf342, (6272, 16, 6), (96, 6, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [matmul_73], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf343, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf344, (6272, 16, 6), (96, 6, 1), 0), out=buf345)
        buf346 = reinterpret_tensor(buf344, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [x_301], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf345, buf346, 602112, grid=grid(602112), stream=stream0)
        buf347 = reinterpret_tensor(buf345, (25088, 24), (24, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (25088, 24), (24, 1), 0), reinterpret_tensor(arg184_1, (24, 24), (1, 24), 0), out=buf347)
        del arg184_1
        buf348 = reinterpret_tensor(buf347, (1568, 16, 24), (384, 24, 1), 0); del buf347  # reuse
        buf352 = reinterpret_tensor(buf346, (1568, 16, 24), (384, 24, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_35, pixel_embed_36, layer_norm_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf348, buf295, buf302, arg163_1, arg185_1, arg186_1, arg187_1, buf352, 25088, 24, grid=grid(25088), stream=stream0)
        del arg163_1
        del arg185_1
        del arg186_1
        del arg187_1
        buf353 = reinterpret_tensor(buf301, (25088, 96), (96, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf352, (25088, 24), (24, 1), 0), reinterpret_tensor(arg188_1, (24, 96), (1, 24), 0), out=buf353)
        del arg188_1
        buf354 = reinterpret_tensor(buf353, (1568, 16, 96), (1536, 96, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [x_305], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf354, arg189_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg189_1
        buf355 = reinterpret_tensor(buf352, (25088, 24), (24, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (25088, 96), (96, 1), 0), reinterpret_tensor(arg190_1, (96, 24), (1, 96), 0), out=buf355)
        del arg190_1
        buf364 = reinterpret_tensor(buf302, (1568, 16, 24), (384, 24, 1), 0); del buf302  # reuse
        buf390 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_37, layer_norm_97, layer_norm_100], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf348, buf355, arg191_1, arg192_1, arg193_1, arg208_1, arg209_1, buf364, buf390, 25088, 24, grid=grid(25088), stream=stream0)
        del arg192_1
        del arg193_1
        del arg208_1
        del arg209_1
        buf360 = reinterpret_tensor(buf308, (1576, 1536), (1536, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf359, (1576, 384), (384, 1), 0), reinterpret_tensor(arg176_1, (384, 1536), (1, 384), 0), out=buf360)
        del arg176_1
        buf361 = reinterpret_tensor(buf360, (8, 197, 1536), (302592, 1536, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [x_297], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf361, arg177_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg177_1
        buf362 = reinterpret_tensor(buf359, (1576, 384), (384, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf361, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg178_1, (1536, 384), (1, 1536), 0), out=buf362)
        del arg178_1
        buf363 = reinterpret_tensor(buf362, (8, 197, 384), (75648, 384, 1), 0); del buf362  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_61, patch_embed_62], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf363, buf312, buf328, arg173_1, arg179_1, 605184, grid=grid(605184), stream=stream0)
        del arg173_1
        del arg179_1
        buf365 = reinterpret_tensor(buf337, (1568, 384), (384, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (1568, 384), (384, 1), 0), reinterpret_tensor(arg194_1, (384, 384), (1, 384), 0), out=buf365)
        del arg194_1
        buf370 = reinterpret_tensor(buf328, (8, 197, 384), (75648, 384, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_63, layer_norm_98], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(buf363, buf365, arg195_1, arg196_1, arg197_1, buf370, 1576, 384, grid=grid(1576), stream=stream0)
        del arg196_1
        del arg197_1
        buf371 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [linear_207], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (1576, 384), (384, 1), 0), reinterpret_tensor(arg198_1, (384, 768), (1, 384), 0), out=buf371)
        del arg198_1
        buf372 = reinterpret_tensor(buf312, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [matmul_74], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf371, buf372, 605184, grid=grid(605184), stream=stream0)
        buf373 = reinterpret_tensor(buf318, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [matmul_74], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf371, buf373, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf374 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [matmul_74], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf372, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf373, (48, 64, 197), (12608, 197, 1), 0), out=buf374)
        buf378 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [attn_112], Original ATen: [aten._softmax]
        triton_red_fused__softmax_18.run(buf374, buf378, 9456, 197, grid=grid(9456), stream=stream0)
        buf377 = reinterpret_tensor(buf373, (1576, 384), (384, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [linear_208], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (1576, 384), (384, 1), 0), reinterpret_tensor(arg199_1, (384, 384), (1, 384), 0), out=buf377)
        del arg199_1
        buf379 = reinterpret_tensor(buf370, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [matmul_75], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf377, buf379, 605184, grid=grid(605184), stream=stream0)
        buf380 = reinterpret_tensor(buf377, (48, 197, 64), (12608, 64, 1), 0); del buf377  # reuse
        # Topologically Sorted Source Nodes: [matmul_75], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf378, (48, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf379, (48, 197, 64), (12608, 64, 1), 0), out=buf380)
        buf381 = reinterpret_tensor(buf379, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [x_309], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf380, buf381, 605184, grid=grid(605184), stream=stream0)
        buf382 = reinterpret_tensor(buf380, (1576, 384), (384, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf381, (1576, 384), (384, 1), 0), reinterpret_tensor(arg200_1, (384, 384), (1, 384), 0), out=buf382)
        del arg200_1
        buf383 = reinterpret_tensor(buf382, (8, 197, 384), (75648, 384, 1), 0); del buf382  # reuse
        buf414 = reinterpret_tensor(buf381, (8, 197, 384), (75648, 384, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_63, patch_embed_64, layer_norm_99], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_21.run(buf383, buf363, buf365, arg195_1, arg201_1, arg202_1, arg203_1, buf414, 1576, 384, grid=grid(1576), stream=stream0)
        del arg195_1
        del arg201_1
        del arg202_1
        del arg203_1
        buf391 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [linear_212], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (25088, 24), (24, 1), 0), reinterpret_tensor(arg210_1, (24, 48), (1, 24), 0), out=buf391)
        del arg210_1
        buf392 = reinterpret_tensor(buf365, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [matmul_76], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf391, buf392, 602112, grid=grid(602112), stream=stream0)
        buf393 = reinterpret_tensor(buf364, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [matmul_76], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf391, buf393, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf394 = reinterpret_tensor(buf343, (6272, 16, 16), (256, 16, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [matmul_76], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf392, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf393, (6272, 6, 16), (96, 16, 1), 0), out=buf394)
        buf398 = reinterpret_tensor(buf339, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf339  # reuse
        # Topologically Sorted Source Nodes: [attn_115], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf394, buf398, 100352, 16, grid=grid(100352), stream=stream0)
        buf397 = reinterpret_tensor(buf393, (25088, 24), (24, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [linear_213], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (25088, 24), (24, 1), 0), reinterpret_tensor(arg211_1, (24, 24), (1, 24), 0), out=buf397)
        del arg211_1
        buf399 = reinterpret_tensor(buf390, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf390  # reuse
        # Topologically Sorted Source Nodes: [matmul_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf397, buf399, 602112, grid=grid(602112), stream=stream0)
        buf400 = reinterpret_tensor(buf397, (6272, 16, 6), (96, 6, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [matmul_77], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf398, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf399, (6272, 16, 6), (96, 6, 1), 0), out=buf400)
        buf401 = reinterpret_tensor(buf399, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [x_317], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf400, buf401, 602112, grid=grid(602112), stream=stream0)
        buf402 = reinterpret_tensor(buf400, (25088, 24), (24, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf401, (25088, 24), (24, 1), 0), reinterpret_tensor(arg212_1, (24, 24), (1, 24), 0), out=buf402)
        del arg212_1
        buf403 = reinterpret_tensor(buf402, (1568, 16, 24), (384, 24, 1), 0); del buf402  # reuse
        buf407 = reinterpret_tensor(buf401, (1568, 16, 24), (384, 24, 1), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_37, pixel_embed_38, layer_norm_101], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf403, buf348, buf355, arg191_1, arg213_1, arg214_1, arg215_1, buf407, 25088, 24, grid=grid(25088), stream=stream0)
        del arg191_1
        del arg213_1
        del arg214_1
        del arg215_1
        buf408 = reinterpret_tensor(buf354, (25088, 96), (96, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf407, (25088, 24), (24, 1), 0), reinterpret_tensor(arg216_1, (24, 96), (1, 24), 0), out=buf408)
        del arg216_1
        buf409 = reinterpret_tensor(buf408, (1568, 16, 96), (1536, 96, 1), 0); del buf408  # reuse
        # Topologically Sorted Source Nodes: [x_321], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf409, arg217_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg217_1
        buf410 = reinterpret_tensor(buf407, (25088, 24), (24, 1), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf409, (25088, 96), (96, 1), 0), reinterpret_tensor(arg218_1, (96, 24), (1, 96), 0), out=buf410)
        del arg218_1
        buf418 = reinterpret_tensor(buf355, (1568, 16, 24), (384, 24, 1), 0); del buf355  # reuse
        buf443 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_39, layer_norm_102, layer_norm_105], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf403, buf410, arg219_1, arg220_1, arg221_1, arg236_1, arg237_1, buf418, buf443, 25088, 24, grid=grid(25088), stream=stream0)
        del arg220_1
        del arg221_1
        del arg236_1
        del arg237_1
        buf415 = reinterpret_tensor(buf361, (1576, 1536), (1536, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf414, (1576, 384), (384, 1), 0), reinterpret_tensor(arg204_1, (384, 1536), (1, 384), 0), out=buf415)
        del arg204_1
        buf416 = reinterpret_tensor(buf415, (8, 197, 1536), (302592, 1536, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [x_313], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf416, arg205_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg205_1
        buf417 = reinterpret_tensor(buf414, (1576, 384), (384, 1), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf416, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg206_1, (1536, 384), (1, 1536), 0), out=buf417)
        del arg206_1
        buf419 = reinterpret_tensor(buf392, (1568, 384), (384, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf418, (1568, 384), (384, 1), 0), reinterpret_tensor(arg222_1, (384, 384), (1, 384), 0), out=buf419)
        del arg222_1
        buf420 = buf363; del buf363  # reuse
        buf424 = reinterpret_tensor(buf372, (8, 197, 384), (75648, 384, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_66, layer_norm_103], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_24.run(buf383, buf417, arg207_1, buf419, arg223_1, arg224_1, arg225_1, buf420, buf424, 1576, 384, grid=grid(1576), stream=stream0)
        del arg207_1
        del arg223_1
        del arg224_1
        del arg225_1
        buf425 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [linear_218], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (1576, 384), (384, 1), 0), reinterpret_tensor(arg226_1, (384, 768), (1, 384), 0), out=buf425)
        del arg226_1
        buf426 = reinterpret_tensor(buf417, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [matmul_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf425, buf426, 605184, grid=grid(605184), stream=stream0)
        buf427 = reinterpret_tensor(buf383, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [matmul_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf425, buf427, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf428 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [matmul_78], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf426, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf427, (48, 64, 197), (12608, 197, 1), 0), out=buf428)
        buf432 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [attn_118], Original ATen: [aten._softmax]
        triton_red_fused__softmax_18.run(buf428, buf432, 9456, 197, grid=grid(9456), stream=stream0)
        buf431 = reinterpret_tensor(buf427, (1576, 384), (384, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [linear_219], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (1576, 384), (384, 1), 0), reinterpret_tensor(arg227_1, (384, 384), (1, 384), 0), out=buf431)
        del arg227_1
        buf433 = reinterpret_tensor(buf424, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [matmul_79], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf431, buf433, 605184, grid=grid(605184), stream=stream0)
        buf434 = reinterpret_tensor(buf431, (48, 197, 64), (12608, 64, 1), 0); del buf431  # reuse
        # Topologically Sorted Source Nodes: [matmul_79], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf432, (48, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf433, (48, 197, 64), (12608, 64, 1), 0), out=buf434)
        buf435 = reinterpret_tensor(buf433, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf433  # reuse
        # Topologically Sorted Source Nodes: [x_325], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf434, buf435, 605184, grid=grid(605184), stream=stream0)
        buf436 = reinterpret_tensor(buf434, (1576, 384), (384, 1), 0); del buf434  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (1576, 384), (384, 1), 0), reinterpret_tensor(arg228_1, (384, 384), (1, 384), 0), out=buf436)
        del arg228_1
        buf467 = reinterpret_tensor(buf435, (8, 197, 384), (75648, 384, 1), 0); del buf435  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_67, layer_norm_104], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf420, buf436, arg229_1, arg230_1, arg231_1, buf467, 1576, 384, grid=grid(1576), stream=stream0)
        del arg230_1
        del arg231_1
        buf444 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [linear_223], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (25088, 24), (24, 1), 0), reinterpret_tensor(arg238_1, (24, 48), (1, 24), 0), out=buf444)
        del arg238_1
        buf445 = reinterpret_tensor(buf419, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [matmul_80], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf444, buf445, 602112, grid=grid(602112), stream=stream0)
        buf446 = reinterpret_tensor(buf418, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [matmul_80], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf444, buf446, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf447 = reinterpret_tensor(buf398, (6272, 16, 16), (256, 16, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [matmul_80], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf445, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf446, (6272, 6, 16), (96, 16, 1), 0), out=buf447)
        buf451 = reinterpret_tensor(buf394, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf394  # reuse
        # Topologically Sorted Source Nodes: [attn_121], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf447, buf451, 100352, 16, grid=grid(100352), stream=stream0)
        buf450 = reinterpret_tensor(buf446, (25088, 24), (24, 1), 0); del buf446  # reuse
        # Topologically Sorted Source Nodes: [linear_224], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (25088, 24), (24, 1), 0), reinterpret_tensor(arg239_1, (24, 24), (1, 24), 0), out=buf450)
        del arg239_1
        buf452 = reinterpret_tensor(buf443, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [matmul_81], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf450, buf452, 602112, grid=grid(602112), stream=stream0)
        buf453 = reinterpret_tensor(buf450, (6272, 16, 6), (96, 6, 1), 0); del buf450  # reuse
        # Topologically Sorted Source Nodes: [matmul_81], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf451, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf452, (6272, 16, 6), (96, 6, 1), 0), out=buf453)
        buf454 = reinterpret_tensor(buf452, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf452  # reuse
        # Topologically Sorted Source Nodes: [x_333], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf453, buf454, 602112, grid=grid(602112), stream=stream0)
        buf455 = reinterpret_tensor(buf453, (25088, 24), (24, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf454, (25088, 24), (24, 1), 0), reinterpret_tensor(arg240_1, (24, 24), (1, 24), 0), out=buf455)
        del arg240_1
        buf456 = reinterpret_tensor(buf455, (1568, 16, 24), (384, 24, 1), 0); del buf455  # reuse
        buf460 = reinterpret_tensor(buf454, (1568, 16, 24), (384, 24, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_39, pixel_embed_40, layer_norm_106], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf456, buf403, buf410, arg219_1, arg241_1, arg242_1, arg243_1, buf460, 25088, 24, grid=grid(25088), stream=stream0)
        del arg219_1
        del arg241_1
        del arg242_1
        del arg243_1
        buf461 = reinterpret_tensor(buf409, (25088, 96), (96, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf460, (25088, 24), (24, 1), 0), reinterpret_tensor(arg244_1, (24, 96), (1, 24), 0), out=buf461)
        del arg244_1
        buf462 = reinterpret_tensor(buf461, (1568, 16, 96), (1536, 96, 1), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [x_337], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf462, arg245_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg245_1
        buf463 = reinterpret_tensor(buf460, (25088, 24), (24, 1), 0); del buf460  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf462, (25088, 96), (96, 1), 0), reinterpret_tensor(arg246_1, (96, 24), (1, 96), 0), out=buf463)
        del arg246_1
        buf472 = reinterpret_tensor(buf410, (1568, 16, 24), (384, 24, 1), 0); del buf410  # reuse
        buf498 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_41, layer_norm_107, layer_norm_110], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf456, buf463, arg247_1, arg248_1, arg249_1, arg264_1, arg265_1, buf472, buf498, 25088, 24, grid=grid(25088), stream=stream0)
        del arg248_1
        del arg249_1
        del arg264_1
        del arg265_1
        buf468 = reinterpret_tensor(buf416, (1576, 1536), (1536, 1), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf467, (1576, 384), (384, 1), 0), reinterpret_tensor(arg232_1, (384, 1536), (1, 384), 0), out=buf468)
        del arg232_1
        buf469 = reinterpret_tensor(buf468, (8, 197, 1536), (302592, 1536, 1), 0); del buf468  # reuse
        # Topologically Sorted Source Nodes: [x_329], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf469, arg233_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg233_1
        buf470 = reinterpret_tensor(buf467, (1576, 384), (384, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf469, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg234_1, (1536, 384), (1, 1536), 0), out=buf470)
        del arg234_1
        buf471 = reinterpret_tensor(buf470, (8, 197, 384), (75648, 384, 1), 0); del buf470  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_67, patch_embed_68], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf471, buf420, buf436, arg229_1, arg235_1, 605184, grid=grid(605184), stream=stream0)
        del arg229_1
        del arg235_1
        buf473 = reinterpret_tensor(buf445, (1568, 384), (384, 1), 0); del buf445  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf472, (1568, 384), (384, 1), 0), reinterpret_tensor(arg250_1, (384, 384), (1, 384), 0), out=buf473)
        del arg250_1
        buf478 = reinterpret_tensor(buf436, (8, 197, 384), (75648, 384, 1), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_69, layer_norm_108], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(buf471, buf473, arg251_1, arg252_1, arg253_1, buf478, 1576, 384, grid=grid(1576), stream=stream0)
        del arg252_1
        del arg253_1
        buf479 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [linear_229], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf478, (1576, 384), (384, 1), 0), reinterpret_tensor(arg254_1, (384, 768), (1, 384), 0), out=buf479)
        del arg254_1
        buf480 = reinterpret_tensor(buf420, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf420  # reuse
        # Topologically Sorted Source Nodes: [matmul_82], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf479, buf480, 605184, grid=grid(605184), stream=stream0)
        buf481 = reinterpret_tensor(buf426, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [matmul_82], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf479, buf481, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf482 = buf428; del buf428  # reuse
        # Topologically Sorted Source Nodes: [matmul_82], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf480, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf481, (48, 64, 197), (12608, 197, 1), 0), out=buf482)
        buf486 = buf432; del buf432  # reuse
        # Topologically Sorted Source Nodes: [attn_124], Original ATen: [aten._softmax]
        triton_red_fused__softmax_18.run(buf482, buf486, 9456, 197, grid=grid(9456), stream=stream0)
        buf485 = reinterpret_tensor(buf481, (1576, 384), (384, 1), 0); del buf481  # reuse
        # Topologically Sorted Source Nodes: [linear_230], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf478, (1576, 384), (384, 1), 0), reinterpret_tensor(arg255_1, (384, 384), (1, 384), 0), out=buf485)
        del arg255_1
        buf487 = reinterpret_tensor(buf478, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [matmul_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf485, buf487, 605184, grid=grid(605184), stream=stream0)
        buf488 = reinterpret_tensor(buf485, (48, 197, 64), (12608, 64, 1), 0); del buf485  # reuse
        # Topologically Sorted Source Nodes: [matmul_83], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf486, (48, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf487, (48, 197, 64), (12608, 64, 1), 0), out=buf488)
        buf489 = reinterpret_tensor(buf487, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf487  # reuse
        # Topologically Sorted Source Nodes: [x_341], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf488, buf489, 605184, grid=grid(605184), stream=stream0)
        buf490 = reinterpret_tensor(buf488, (1576, 384), (384, 1), 0); del buf488  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf489, (1576, 384), (384, 1), 0), reinterpret_tensor(arg256_1, (384, 384), (1, 384), 0), out=buf490)
        del arg256_1
        buf491 = reinterpret_tensor(buf490, (8, 197, 384), (75648, 384, 1), 0); del buf490  # reuse
        buf522 = reinterpret_tensor(buf489, (8, 197, 384), (75648, 384, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_69, patch_embed_70, layer_norm_109], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_21.run(buf491, buf471, buf473, arg251_1, arg257_1, arg258_1, arg259_1, buf522, 1576, 384, grid=grid(1576), stream=stream0)
        del arg251_1
        del arg257_1
        del arg258_1
        del arg259_1
        buf499 = buf444; del buf444  # reuse
        # Topologically Sorted Source Nodes: [linear_234], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf498, (25088, 24), (24, 1), 0), reinterpret_tensor(arg266_1, (24, 48), (1, 24), 0), out=buf499)
        del arg266_1
        buf500 = reinterpret_tensor(buf473, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf473  # reuse
        # Topologically Sorted Source Nodes: [matmul_84], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf499, buf500, 602112, grid=grid(602112), stream=stream0)
        buf501 = reinterpret_tensor(buf472, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf472  # reuse
        # Topologically Sorted Source Nodes: [matmul_84], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf499, buf501, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf502 = reinterpret_tensor(buf451, (6272, 16, 16), (256, 16, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [matmul_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf500, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf501, (6272, 6, 16), (96, 16, 1), 0), out=buf502)
        buf506 = reinterpret_tensor(buf447, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf447  # reuse
        # Topologically Sorted Source Nodes: [attn_127], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf502, buf506, 100352, 16, grid=grid(100352), stream=stream0)
        buf505 = reinterpret_tensor(buf501, (25088, 24), (24, 1), 0); del buf501  # reuse
        # Topologically Sorted Source Nodes: [linear_235], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf498, (25088, 24), (24, 1), 0), reinterpret_tensor(arg267_1, (24, 24), (1, 24), 0), out=buf505)
        del arg267_1
        buf507 = reinterpret_tensor(buf498, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf498  # reuse
        # Topologically Sorted Source Nodes: [matmul_85], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf505, buf507, 602112, grid=grid(602112), stream=stream0)
        buf508 = reinterpret_tensor(buf505, (6272, 16, 6), (96, 6, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [matmul_85], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf506, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf507, (6272, 16, 6), (96, 6, 1), 0), out=buf508)
        buf509 = reinterpret_tensor(buf507, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf507  # reuse
        # Topologically Sorted Source Nodes: [x_349], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf508, buf509, 602112, grid=grid(602112), stream=stream0)
        buf510 = reinterpret_tensor(buf508, (25088, 24), (24, 1), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf509, (25088, 24), (24, 1), 0), reinterpret_tensor(arg268_1, (24, 24), (1, 24), 0), out=buf510)
        del arg268_1
        buf511 = reinterpret_tensor(buf510, (1568, 16, 24), (384, 24, 1), 0); del buf510  # reuse
        buf515 = reinterpret_tensor(buf509, (1568, 16, 24), (384, 24, 1), 0); del buf509  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_41, pixel_embed_42, layer_norm_111], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf511, buf456, buf463, arg247_1, arg269_1, arg270_1, arg271_1, buf515, 25088, 24, grid=grid(25088), stream=stream0)
        del arg247_1
        del arg269_1
        del arg270_1
        del arg271_1
        buf516 = reinterpret_tensor(buf462, (25088, 96), (96, 1), 0); del buf462  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf515, (25088, 24), (24, 1), 0), reinterpret_tensor(arg272_1, (24, 96), (1, 24), 0), out=buf516)
        del arg272_1
        buf517 = reinterpret_tensor(buf516, (1568, 16, 96), (1536, 96, 1), 0); del buf516  # reuse
        # Topologically Sorted Source Nodes: [x_353], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf517, arg273_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg273_1
        buf518 = reinterpret_tensor(buf515, (25088, 24), (24, 1), 0); del buf515  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf517, (25088, 96), (96, 1), 0), reinterpret_tensor(arg274_1, (96, 24), (1, 96), 0), out=buf518)
        del arg274_1
        buf526 = reinterpret_tensor(buf463, (1568, 16, 24), (384, 24, 1), 0); del buf463  # reuse
        buf551 = buf456; del buf456  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_43, layer_norm_112, layer_norm_115], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf511, buf518, arg275_1, arg276_1, arg277_1, arg292_1, arg293_1, buf526, buf551, 25088, 24, grid=grid(25088), stream=stream0)
        del arg276_1
        del arg277_1
        del arg292_1
        del arg293_1
        buf523 = reinterpret_tensor(buf469, (1576, 1536), (1536, 1), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf522, (1576, 384), (384, 1), 0), reinterpret_tensor(arg260_1, (384, 1536), (1, 384), 0), out=buf523)
        del arg260_1
        buf524 = reinterpret_tensor(buf523, (8, 197, 1536), (302592, 1536, 1), 0); del buf523  # reuse
        # Topologically Sorted Source Nodes: [x_345], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf524, arg261_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg261_1
        buf525 = reinterpret_tensor(buf522, (1576, 384), (384, 1), 0); del buf522  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf524, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg262_1, (1536, 384), (1, 1536), 0), out=buf525)
        del arg262_1
        buf527 = reinterpret_tensor(buf500, (1568, 384), (384, 1), 0); del buf500  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf526, (1568, 384), (384, 1), 0), reinterpret_tensor(arg278_1, (384, 384), (1, 384), 0), out=buf527)
        del arg278_1
        buf528 = buf471; del buf471  # reuse
        buf532 = reinterpret_tensor(buf480, (8, 197, 384), (75648, 384, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_72, layer_norm_113], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_24.run(buf491, buf525, arg263_1, buf527, arg279_1, arg280_1, arg281_1, buf528, buf532, 1576, 384, grid=grid(1576), stream=stream0)
        del arg263_1
        del arg279_1
        del arg280_1
        del arg281_1
        buf533 = buf479; del buf479  # reuse
        # Topologically Sorted Source Nodes: [linear_240], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf532, (1576, 384), (384, 1), 0), reinterpret_tensor(arg282_1, (384, 768), (1, 384), 0), out=buf533)
        del arg282_1
        buf534 = reinterpret_tensor(buf525, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf525  # reuse
        # Topologically Sorted Source Nodes: [matmul_86], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf533, buf534, 605184, grid=grid(605184), stream=stream0)
        buf535 = reinterpret_tensor(buf491, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf491  # reuse
        # Topologically Sorted Source Nodes: [matmul_86], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf533, buf535, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf536 = buf482; del buf482  # reuse
        # Topologically Sorted Source Nodes: [matmul_86], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf534, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf535, (48, 64, 197), (12608, 197, 1), 0), out=buf536)
        buf540 = buf486; del buf486  # reuse
        # Topologically Sorted Source Nodes: [attn_130], Original ATen: [aten._softmax]
        triton_red_fused__softmax_18.run(buf536, buf540, 9456, 197, grid=grid(9456), stream=stream0)
        buf539 = reinterpret_tensor(buf535, (1576, 384), (384, 1), 0); del buf535  # reuse
        # Topologically Sorted Source Nodes: [linear_241], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf532, (1576, 384), (384, 1), 0), reinterpret_tensor(arg283_1, (384, 384), (1, 384), 0), out=buf539)
        del arg283_1
        buf541 = reinterpret_tensor(buf532, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [matmul_87], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf539, buf541, 605184, grid=grid(605184), stream=stream0)
        buf542 = reinterpret_tensor(buf539, (48, 197, 64), (12608, 64, 1), 0); del buf539  # reuse
        # Topologically Sorted Source Nodes: [matmul_87], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf540, (48, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf541, (48, 197, 64), (12608, 64, 1), 0), out=buf542)
        buf543 = reinterpret_tensor(buf541, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf541  # reuse
        # Topologically Sorted Source Nodes: [x_357], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf542, buf543, 605184, grid=grid(605184), stream=stream0)
        buf544 = reinterpret_tensor(buf542, (1576, 384), (384, 1), 0); del buf542  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf543, (1576, 384), (384, 1), 0), reinterpret_tensor(arg284_1, (384, 384), (1, 384), 0), out=buf544)
        del arg284_1
        buf575 = reinterpret_tensor(buf543, (8, 197, 384), (75648, 384, 1), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_73, layer_norm_114], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf528, buf544, arg285_1, arg286_1, arg287_1, buf575, 1576, 384, grid=grid(1576), stream=stream0)
        del arg286_1
        del arg287_1
        buf552 = buf499; del buf499  # reuse
        # Topologically Sorted Source Nodes: [linear_245], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf551, (25088, 24), (24, 1), 0), reinterpret_tensor(arg294_1, (24, 48), (1, 24), 0), out=buf552)
        del arg294_1
        buf553 = reinterpret_tensor(buf527, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf527  # reuse
        # Topologically Sorted Source Nodes: [matmul_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf552, buf553, 602112, grid=grid(602112), stream=stream0)
        buf554 = reinterpret_tensor(buf526, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf526  # reuse
        # Topologically Sorted Source Nodes: [matmul_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf552, buf554, 37632, 16, grid=grid(37632, 16), stream=stream0)
        buf555 = reinterpret_tensor(buf506, (6272, 16, 16), (256, 16, 1), 0); del buf506  # reuse
        # Topologically Sorted Source Nodes: [matmul_88], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf553, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf554, (6272, 6, 16), (96, 16, 1), 0), out=buf555)
        buf559 = reinterpret_tensor(buf502, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [attn_133], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf555, buf559, 100352, 16, grid=grid(100352), stream=stream0)
        buf558 = reinterpret_tensor(buf554, (25088, 24), (24, 1), 0); del buf554  # reuse
        # Topologically Sorted Source Nodes: [linear_246], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf551, (25088, 24), (24, 1), 0), reinterpret_tensor(arg295_1, (24, 24), (1, 24), 0), out=buf558)
        del arg295_1
        buf560 = reinterpret_tensor(buf551, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf551  # reuse
        # Topologically Sorted Source Nodes: [matmul_89], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf558, buf560, 602112, grid=grid(602112), stream=stream0)
        buf561 = reinterpret_tensor(buf558, (6272, 16, 6), (96, 6, 1), 0); del buf558  # reuse
        # Topologically Sorted Source Nodes: [matmul_89], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf559, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf560, (6272, 16, 6), (96, 6, 1), 0), out=buf561)
        buf562 = reinterpret_tensor(buf560, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf560  # reuse
        # Topologically Sorted Source Nodes: [x_365], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf561, buf562, 602112, grid=grid(602112), stream=stream0)
        buf563 = reinterpret_tensor(buf561, (25088, 24), (24, 1), 0); del buf561  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf562, (25088, 24), (24, 1), 0), reinterpret_tensor(arg296_1, (24, 24), (1, 24), 0), out=buf563)
        del arg296_1
        buf564 = reinterpret_tensor(buf563, (1568, 16, 24), (384, 24, 1), 0); del buf563  # reuse
        buf568 = reinterpret_tensor(buf562, (1568, 16, 24), (384, 24, 1), 0); del buf562  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_43, pixel_embed_44, layer_norm_116], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf564, buf511, buf518, arg275_1, arg297_1, arg298_1, arg299_1, buf568, 25088, 24, grid=grid(25088), stream=stream0)
        del arg275_1
        del arg297_1
        del arg298_1
        del arg299_1
        buf569 = reinterpret_tensor(buf517, (25088, 96), (96, 1), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf568, (25088, 24), (24, 1), 0), reinterpret_tensor(arg300_1, (24, 96), (1, 24), 0), out=buf569)
        del arg300_1
        buf570 = reinterpret_tensor(buf569, (1568, 16, 96), (1536, 96, 1), 0); del buf569  # reuse
        # Topologically Sorted Source Nodes: [x_369], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf570, arg301_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg301_1
        buf571 = reinterpret_tensor(buf568, (25088, 24), (24, 1), 0); del buf568  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf570, (25088, 96), (96, 1), 0), reinterpret_tensor(arg302_1, (96, 24), (1, 96), 0), out=buf571)
        del arg302_1
        buf580 = reinterpret_tensor(buf518, (1568, 16, 24), (384, 24, 1), 0); del buf518  # reuse
        buf606 = buf511; del buf511  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_45, layer_norm_117, layer_norm_120], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf564, buf571, arg303_1, arg304_1, arg305_1, arg320_1, arg321_1, buf580, buf606, 25088, 24, grid=grid(25088), stream=stream0)
        del arg304_1
        del arg305_1
        del arg320_1
        del arg321_1
        buf576 = reinterpret_tensor(buf524, (1576, 1536), (1536, 1), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf575, (1576, 384), (384, 1), 0), reinterpret_tensor(arg288_1, (384, 1536), (1, 384), 0), out=buf576)
        del arg288_1
        buf577 = reinterpret_tensor(buf576, (8, 197, 1536), (302592, 1536, 1), 0); del buf576  # reuse
        # Topologically Sorted Source Nodes: [x_361], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf577, arg289_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg289_1
        buf578 = reinterpret_tensor(buf575, (1576, 384), (384, 1), 0); del buf575  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf577, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg290_1, (1536, 384), (1, 1536), 0), out=buf578)
        del arg290_1
        buf579 = reinterpret_tensor(buf578, (8, 197, 384), (75648, 384, 1), 0); del buf578  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_73, patch_embed_74], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf579, buf528, buf544, arg285_1, arg291_1, 605184, grid=grid(605184), stream=stream0)
        del arg285_1
        del arg291_1
        buf581 = reinterpret_tensor(buf553, (1568, 384), (384, 1), 0); del buf553  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf580, (1568, 384), (384, 1), 0), reinterpret_tensor(arg306_1, (384, 384), (1, 384), 0), out=buf581)
        del arg306_1
        buf586 = reinterpret_tensor(buf544, (8, 197, 384), (75648, 384, 1), 0); del buf544  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_75, layer_norm_118], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(buf579, buf581, arg307_1, arg308_1, arg309_1, buf586, 1576, 384, grid=grid(1576), stream=stream0)
        del arg308_1
        del arg309_1
        buf587 = buf533; del buf533  # reuse
        # Topologically Sorted Source Nodes: [linear_251], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf586, (1576, 384), (384, 1), 0), reinterpret_tensor(arg310_1, (384, 768), (1, 384), 0), out=buf587)
        del arg310_1
        buf588 = reinterpret_tensor(buf528, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf528  # reuse
        # Topologically Sorted Source Nodes: [matmul_90], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf587, buf588, 605184, grid=grid(605184), stream=stream0)
        buf589 = reinterpret_tensor(buf534, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf534  # reuse
        # Topologically Sorted Source Nodes: [matmul_90], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf587, buf589, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf590 = buf536; del buf536  # reuse
        # Topologically Sorted Source Nodes: [matmul_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf588, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf589, (48, 64, 197), (12608, 197, 1), 0), out=buf590)
        buf594 = buf540; del buf540  # reuse
        # Topologically Sorted Source Nodes: [attn_136], Original ATen: [aten._softmax]
        triton_red_fused__softmax_18.run(buf590, buf594, 9456, 197, grid=grid(9456), stream=stream0)
        buf593 = reinterpret_tensor(buf589, (1576, 384), (384, 1), 0); del buf589  # reuse
        # Topologically Sorted Source Nodes: [linear_252], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf586, (1576, 384), (384, 1), 0), reinterpret_tensor(arg311_1, (384, 384), (1, 384), 0), out=buf593)
        del arg311_1
        buf595 = reinterpret_tensor(buf586, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf586  # reuse
        # Topologically Sorted Source Nodes: [matmul_91], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf593, buf595, 605184, grid=grid(605184), stream=stream0)
        buf596 = reinterpret_tensor(buf593, (48, 197, 64), (12608, 64, 1), 0); del buf593  # reuse
        # Topologically Sorted Source Nodes: [matmul_91], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf594, (48, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf595, (48, 197, 64), (12608, 64, 1), 0), out=buf596)
        buf597 = reinterpret_tensor(buf595, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf595  # reuse
        # Topologically Sorted Source Nodes: [x_373], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf596, buf597, 605184, grid=grid(605184), stream=stream0)
        buf598 = reinterpret_tensor(buf596, (1576, 384), (384, 1), 0); del buf596  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf597, (1576, 384), (384, 1), 0), reinterpret_tensor(arg312_1, (384, 384), (1, 384), 0), out=buf598)
        del arg312_1
        buf599 = reinterpret_tensor(buf598, (8, 197, 384), (75648, 384, 1), 0); del buf598  # reuse
        buf630 = reinterpret_tensor(buf597, (8, 197, 384), (75648, 384, 1), 0); del buf597  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_75, patch_embed_76, layer_norm_119], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_21.run(buf599, buf579, buf581, arg307_1, arg313_1, arg314_1, arg315_1, buf630, 1576, 384, grid=grid(1576), stream=stream0)
        del arg307_1
        del arg313_1
        del arg314_1
        del arg315_1
        buf607 = buf552; del buf552  # reuse
        # Topologically Sorted Source Nodes: [linear_256], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf606, (25088, 24), (24, 1), 0), reinterpret_tensor(arg322_1, (24, 48), (1, 24), 0), out=buf607)
        del arg322_1
        buf608 = reinterpret_tensor(buf581, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf581  # reuse
        # Topologically Sorted Source Nodes: [matmul_92], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf607, buf608, 602112, grid=grid(602112), stream=stream0)
        buf609 = reinterpret_tensor(buf580, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf580  # reuse
        # Topologically Sorted Source Nodes: [matmul_92], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf607, buf609, 37632, 16, grid=grid(37632, 16), stream=stream0)
        del buf607
        buf610 = reinterpret_tensor(buf559, (6272, 16, 16), (256, 16, 1), 0); del buf559  # reuse
        # Topologically Sorted Source Nodes: [matmul_92], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf608, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf609, (6272, 6, 16), (96, 16, 1), 0), out=buf610)
        del buf608
        buf614 = reinterpret_tensor(buf555, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf555  # reuse
        # Topologically Sorted Source Nodes: [attn_139], Original ATen: [aten._softmax]
        triton_per_fused__softmax_8.run(buf610, buf614, 100352, 16, grid=grid(100352), stream=stream0)
        del buf610
        buf613 = reinterpret_tensor(buf609, (25088, 24), (24, 1), 0); del buf609  # reuse
        # Topologically Sorted Source Nodes: [linear_257], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf606, (25088, 24), (24, 1), 0), reinterpret_tensor(arg323_1, (24, 24), (1, 24), 0), out=buf613)
        del arg323_1
        buf615 = reinterpret_tensor(buf606, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf606  # reuse
        # Topologically Sorted Source Nodes: [matmul_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf613, buf615, 602112, grid=grid(602112), stream=stream0)
        buf616 = reinterpret_tensor(buf613, (6272, 16, 6), (96, 6, 1), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [matmul_93], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf614, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf615, (6272, 16, 6), (96, 6, 1), 0), out=buf616)
        del buf614
        buf617 = reinterpret_tensor(buf615, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf615  # reuse
        # Topologically Sorted Source Nodes: [x_381], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf616, buf617, 602112, grid=grid(602112), stream=stream0)
        buf618 = reinterpret_tensor(buf616, (25088, 24), (24, 1), 0); del buf616  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf617, (25088, 24), (24, 1), 0), reinterpret_tensor(arg324_1, (24, 24), (1, 24), 0), out=buf618)
        del arg324_1
        buf619 = reinterpret_tensor(buf618, (1568, 16, 24), (384, 24, 1), 0); del buf618  # reuse
        buf623 = reinterpret_tensor(buf617, (1568, 16, 24), (384, 24, 1), 0); del buf617  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_45, pixel_embed_46, layer_norm_121], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf619, buf564, buf571, arg303_1, arg325_1, arg326_1, arg327_1, buf623, 25088, 24, grid=grid(25088), stream=stream0)
        del arg303_1
        del arg325_1
        del arg326_1
        del arg327_1
        del buf564
        buf624 = reinterpret_tensor(buf570, (25088, 96), (96, 1), 0); del buf570  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf623, (25088, 24), (24, 1), 0), reinterpret_tensor(arg328_1, (24, 96), (1, 24), 0), out=buf624)
        del arg328_1
        buf625 = reinterpret_tensor(buf624, (1568, 16, 96), (1536, 96, 1), 0); del buf624  # reuse
        # Topologically Sorted Source Nodes: [x_385], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf625, arg329_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg329_1
        buf626 = reinterpret_tensor(buf623, (25088, 24), (24, 1), 0); del buf623  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf625, (25088, 96), (96, 1), 0), reinterpret_tensor(arg330_1, (96, 24), (1, 96), 0), out=buf626)
        del arg330_1
        del buf625
        buf634 = reinterpret_tensor(buf571, (1568, 16, 24), (384, 24, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [pixel_embed_47, layer_norm_122], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf619, buf626, arg331_1, arg332_1, arg333_1, buf634, 25088, 24, grid=grid(25088), stream=stream0)
        del arg331_1
        del arg332_1
        del arg333_1
        del buf619
        buf631 = reinterpret_tensor(buf577, (1576, 1536), (1536, 1), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf630, (1576, 384), (384, 1), 0), reinterpret_tensor(arg316_1, (384, 1536), (1, 384), 0), out=buf631)
        del arg316_1
        buf632 = reinterpret_tensor(buf631, (8, 197, 1536), (302592, 1536, 1), 0); del buf631  # reuse
        # Topologically Sorted Source Nodes: [x_377], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf632, arg317_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg317_1
        buf633 = reinterpret_tensor(buf630, (1576, 384), (384, 1), 0); del buf630  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf632, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg318_1, (1536, 384), (1, 1536), 0), out=buf633)
        del arg318_1
        buf635 = reinterpret_tensor(buf626, (1568, 384), (384, 1), 0); del buf626  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf634, (1568, 384), (384, 1), 0), reinterpret_tensor(arg334_1, (384, 384), (1, 384), 0), out=buf635)
        del arg334_1
        del buf634
        buf636 = buf579; del buf579  # reuse
        buf640 = reinterpret_tensor(buf588, (8, 197, 384), (75648, 384, 1), 0); del buf588  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_78, layer_norm_123], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_24.run(buf599, buf633, arg319_1, buf635, arg335_1, arg336_1, arg337_1, buf636, buf640, 1576, 384, grid=grid(1576), stream=stream0)
        del arg319_1
        del arg335_1
        del arg336_1
        del arg337_1
        del buf635
        buf641 = buf587; del buf587  # reuse
        # Topologically Sorted Source Nodes: [linear_262], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf640, (1576, 384), (384, 1), 0), reinterpret_tensor(arg338_1, (384, 768), (1, 384), 0), out=buf641)
        del arg338_1
        buf642 = reinterpret_tensor(buf633, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf633  # reuse
        # Topologically Sorted Source Nodes: [matmul_94], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf641, buf642, 605184, grid=grid(605184), stream=stream0)
        buf643 = reinterpret_tensor(buf599, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf599  # reuse
        # Topologically Sorted Source Nodes: [matmul_94], Original ATen: [aten.clone]
        triton_poi_fused_clone_17.run(buf641, buf643, 3072, 197, grid=grid(3072, 197), stream=stream0)
        del buf641
        buf644 = buf590; del buf590  # reuse
        # Topologically Sorted Source Nodes: [matmul_94], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf642, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf643, (48, 64, 197), (12608, 197, 1), 0), out=buf644)
        del buf642
        buf648 = buf594; del buf594  # reuse
        # Topologically Sorted Source Nodes: [attn_142], Original ATen: [aten._softmax]
        triton_red_fused__softmax_18.run(buf644, buf648, 9456, 197, grid=grid(9456), stream=stream0)
        del buf644
        buf647 = reinterpret_tensor(buf643, (1576, 384), (384, 1), 0); del buf643  # reuse
        # Topologically Sorted Source Nodes: [linear_263], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf640, (1576, 384), (384, 1), 0), reinterpret_tensor(arg339_1, (384, 384), (1, 384), 0), out=buf647)
        del arg339_1
        buf649 = reinterpret_tensor(buf640, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf640  # reuse
        # Topologically Sorted Source Nodes: [matmul_95], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf647, buf649, 605184, grid=grid(605184), stream=stream0)
        buf650 = reinterpret_tensor(buf647, (48, 197, 64), (12608, 64, 1), 0); del buf647  # reuse
        # Topologically Sorted Source Nodes: [matmul_95], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf648, (48, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf649, (48, 197, 64), (12608, 64, 1), 0), out=buf650)
        del buf648
        buf651 = reinterpret_tensor(buf649, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf649  # reuse
        # Topologically Sorted Source Nodes: [x_389], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf650, buf651, 605184, grid=grid(605184), stream=stream0)
        buf652 = reinterpret_tensor(buf650, (1576, 384), (384, 1), 0); del buf650  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf651, (1576, 384), (384, 1), 0), reinterpret_tensor(arg340_1, (384, 384), (1, 384), 0), out=buf652)
        del arg340_1
        buf656 = reinterpret_tensor(buf651, (8, 197, 384), (75648, 384, 1), 0); del buf651  # reuse
        # Topologically Sorted Source Nodes: [patch_embed_79, layer_norm_124], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_25.run(buf636, buf652, arg341_1, arg342_1, arg343_1, buf656, 1576, 384, grid=grid(1576), stream=stream0)
        del arg342_1
        del arg343_1
        buf657 = reinterpret_tensor(buf632, (1576, 1536), (1536, 1), 0); del buf632  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf656, (1576, 384), (384, 1), 0), reinterpret_tensor(arg344_1, (384, 1536), (1, 384), 0), out=buf657)
        del arg344_1
        buf658 = reinterpret_tensor(buf657, (8, 197, 1536), (302592, 1536, 1), 0); del buf657  # reuse
        # Topologically Sorted Source Nodes: [x_393], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf658, arg345_1, 2420736, grid=grid(2420736), stream=stream0)
        del arg345_1
        buf659 = reinterpret_tensor(buf656, (1576, 384), (384, 1), 0); del buf656  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf658, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg346_1, (1536, 384), (1, 1536), 0), out=buf659)
        del arg346_1
        del buf658
        buf660 = reinterpret_tensor(buf659, (8, 197, 384), (75648, 384, 1), 0); del buf659  # reuse
        buf661 = empty_strided_cuda((8, 197, 1), (197, 1, 1600), torch.float32)
        buf662 = empty_strided_cuda((8, 197, 1), (197, 1, 1600), torch.float32)
        # Topologically Sorted Source Nodes: [patch_embed_79, patch_embed_80, patch_embed_81], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_28.run(buf660, buf636, buf652, arg341_1, arg347_1, buf661, buf662, 1576, 384, grid=grid(1576), stream=stream0)
        del arg341_1
        del arg347_1
        del buf636
        del buf652
        buf664 = empty_strided_cuda((8, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_398], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf660, buf661, buf662, arg348_1, arg349_1, buf664, 3072, grid=grid(3072), stream=stream0)
        del arg348_1
        del arg349_1
        del buf660
        del buf661
        del buf662
        buf665 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_398, x_399], Original ATen: [aten.clone, aten.addmm]
        extern_kernels.addmm(arg351_1, buf664, reinterpret_tensor(arg350_1, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf665)
        del arg350_1
        del arg351_1
        del buf664
    return (buf665, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 24, 4, 4), (384, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((24, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((48, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((24, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((96, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((24, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tnt_s_patch16_224', benchmark_compiled_module)
