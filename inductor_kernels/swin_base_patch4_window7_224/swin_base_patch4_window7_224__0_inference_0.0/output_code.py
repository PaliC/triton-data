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


# kernel path: /tmp/torchinductor_sahanp/qp/cqpvgzqlgafrtw5uva5ergqegwqyokr2h5ytjqx37562y7ic5c7v.py
# Topologically Sorted Source Nodes: [x_416], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_416 => clone_265, var_mean_53
# Graph fragment:
#   %clone_265 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_248,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_53 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_265, [3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_red_fused_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(out_ptr1 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/c7/cc7enay4fow4enfo2amyot6rlpu26kofdrwtr4be5m7rxdhe46q7.py
# Topologically Sorted Source Nodes: [x_416], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_416 => add_257, add_258, clone_265, mul_202, mul_203, rsqrt_53, sub_77, var_mean_53
# Graph fragment:
#   %clone_265 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_248,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_53 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_265, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_265, %getitem_179), kwargs = {})
#   %add_257 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_178, 1e-05), kwargs = {})
#   %rsqrt_53 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_257,), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %rsqrt_53), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %arg3_1), kwargs = {})
#   %add_258 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %arg4_1), kwargs = {})
triton_poi_fused_native_layer_norm_1 = async_compile.triton('triton_poi_fused_native_layer_norm_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 3136) % 128
    x0 = xindex % 3136
    x2 = (xindex // 401408)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (3136*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (3136*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 128.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mg/cmggpran4xidihqsn7t3unxfpsbiy6rjqgm3s3yssdxf5knhnkwr.py
# Topologically Sorted Source Nodes: [shifted_x_72, contiguous_96], Original ATen: [aten.native_layer_norm, aten.clone]
# Source node to ATen node mapping:
#   contiguous_96 => clone_266
#   shifted_x_72 => var_mean_54
# Graph fragment:
#   %var_mean_54 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_258, [3]), kwargs = {correction: 0, keepdim: True})
#   %clone_266 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_249,), kwargs = {memory_format: torch.contiguous_format})
triton_red_fused_clone_native_layer_norm_2 = async_compile.triton('triton_red_fused_clone_native_layer_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_clone_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x7 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    x3 = xindex % 7
    x4 = (xindex // 7) % 8
    x5 = (xindex // 56) % 7
    x8 = (xindex // 392)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 128.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tl.store(out_ptr2 + (r2 + (128*x3) + (896*x5) + (6272*x4) + (50176*x8)), tmp16, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hj/chjx7gfddeis2xikqlo2hthlwuthy5uxlf4ipbtjkphvffvjvcbs.py
# Topologically Sorted Source Nodes: [q_49, attn_118], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   attn_118 => clone_267
#   q_49 => mul_206
# Graph fragment:
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_182, 0.1767766952966369), kwargs = {})
#   %clone_267 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_96,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_3 = async_compile.triton('triton_poi_fused_clone_mul_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 4
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (384*x1) + (18816*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ft/cftabfi3ew23q7ai5yljss6ogbamz56grzzxiyuwiqwz7nbbhg2k.py
# Topologically Sorted Source Nodes: [attn_118], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_118 => clone_268
# Graph fragment:
#   %clone_268 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_97,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 49
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (128 + y0 + (384*x2) + (18816*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (128 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/de/cdehbusdhfjq7uaezyy565o4zgcin3cgtvtjftewi5ayq6wkay3u.py
# Topologically Sorted Source Nodes: [attn_119, attn_120], Original ATen: [aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_119 => add_261
#   attn_120 => amax_24, div_24, exp_24, sub_79, sum_25
# Graph fragment:
#   %add_261 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_666, %unsqueeze_46), kwargs = {})
#   %amax_24 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_261, [-1], True), kwargs = {})
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_261, %amax_24), kwargs = {})
#   %exp_24 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_79,), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [-1], True), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_24, %sum_25), kwargs = {})
triton_per_fused__softmax_add_5 = async_compile.triton('triton_per_fused__softmax_add_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_5(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 4
    x5 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.full([XBLOCK, RBLOCK], 169, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 169)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp5 < 169")
    tmp7 = tl.load(in_ptr2 + (x1 + (4*tmp5)), rmask & xmask, eviction_policy='evict_last')
    tmp8 = tmp0 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr2 + (r3 + (49*x0) + (2432*x5)), tmp19, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vu/cvullsbvr4ihvyzhzw3oe23bohzep2kcidgeesk5fc5dgtflgar2.py
# Topologically Sorted Source Nodes: [x_418], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_418 => clone_271
# Graph fragment:
#   %clone_271 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_99,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 4
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (32*x2) + (384*x1) + (18816*x3)), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cl/cclu5zx3gwy4mdxkuk6cy4ka2bwsnsxo4kat5z5mr7qf423cejk6.py
# Topologically Sorted Source Nodes: [x_419], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_419 => clone_272
# Graph fragment:
#   %clone_272 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_254,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 4
    x2 = (xindex // 128) % 49
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1568*x1) + (6272*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zf/czff5uebe7sz2vdf7q4izxz276qlmep5sno4rtqxq7qf3nkkrbs5.py
# Topologically Sorted Source Nodes: [layer_norm_55], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_55 => add_263, add_264, mul_207, mul_208, rsqrt_55, sub_80, var_mean_55
# Graph fragment:
#   %var_mean_55 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_678, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_678, %getitem_186), kwargs = {})
#   %add_263 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_185, 1e-05), kwargs = {})
#   %rsqrt_55 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_263,), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %rsqrt_55), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_207, %arg13_1), kwargs = {})
#   %add_264 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_208, %arg14_1), kwargs = {})
triton_red_fused_native_layer_norm_8 = async_compile.triton('triton_red_fused_native_layer_norm_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*((x0 % 56) % 7)) + (896*((x0 // 56) % 7)) + (6272*((x0 % 56) // 7)) + (50176*(x0 // 392)) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
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
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r2 + (128*((x0 % 56) % 7)) + (896*((x0 // 56) % 7)) + (6272*((x0 % 56) // 7)) + (50176*(x0 // 392)) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp9 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 128.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-05
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/y3/cy343sxq7qh2tj7kx46pscjdw5dkmwnw3zbwy7grwcyohpu23ejh.py
# Topologically Sorted Source Nodes: [x_427], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_427 => add_265, erf_24, mul_209, mul_210, mul_211
# Graph fragment:
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_680, 0.5), kwargs = {})
#   %mul_210 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_680, 0.7071067811865476), kwargs = {})
#   %erf_24 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_210,), kwargs = {})
#   %add_265 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_24, 1), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_209, %add_265), kwargs = {})
triton_poi_fused_gelu_9 = async_compile.triton('triton_poi_fused_gelu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/4q/c4qvwwxrjiwpg6nmzm7oofrj33w2ooklinuzlf5xfpdizboolzjm.py
# Topologically Sorted Source Nodes: [x_431, layer_norm_56], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_56 => var_mean_56
#   x_431 => add_266
# Graph fragment:
#   %add_266 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_678, %view_682), kwargs = {})
#   %var_mean_56 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_683, [3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_10 = async_compile.triton('triton_red_fused_add_native_layer_norm_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x3 = xindex
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*((x0 % 56) % 7)) + (896*((x0 // 56) % 7)) + (6272*((x0 % 56) // 7)) + (50176*(x0 // 392)) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zt/cztnyscqomoksgfvtw6av5vtgbzapwqql36qnraj76p2ahekmruk.py
# Topologically Sorted Source Nodes: [contiguous_100], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_100 => clone_277
# Graph fragment:
#   %clone_277 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_259,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_11 = async_compile.triton('triton_poi_fused_clone_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x5 = (xindex // 401408)
    x6 = (xindex // 128) % 56
    x7 = (xindex // 7168) % 56
    x2 = (xindex // 896) % 8
    x3 = (xindex // 7168) % 7
    x8 = xindex % 896
    x9 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x0 + (128*((3 + x6) % 56)) + (7168*((3 + x7) % 56)) + (401408*x5)), None)
    tmp1 = tl.load(in_ptr1 + ((56*((3 + x7) % 56)) + (3136*x5) + ((3 + x6) % 56)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((56*((3 + x7) % 56)) + (3136*x5) + ((3 + x6) % 56)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x8 + (896*x3) + (6272*x2) + (50176*x9)), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mx/cmxoa5m2lq7x773q6wdx3ocl6q6fcelxccawjc6uwyqszwmqmxfc.py
# Topologically Sorted Source Nodes: [attn_126], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_126 => amax_25, div_25, exp_25, sub_82, sum_26
# Graph fragment:
#   %amax_25 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_696, [-1], True), kwargs = {})
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_696, %amax_25), kwargs = {})
#   %exp_25 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_82,), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_25, [-1], True), kwargs = {})
#   %div_25 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_25, %sum_26), kwargs = {})
triton_per_fused__softmax_12 = async_compile.triton('triton_per_fused__softmax_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x5 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 4
    x2 = (xindex // 196)
    x4 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x5)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r3 + (49*x0) + (2401*(x2 % 64))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.full([XBLOCK, RBLOCK], 169, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 169)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp5 < 169")
    tmp7 = tl.load(in_ptr2 + (x1 + (4*tmp5)), rmask & xmask, eviction_policy='evict_last')
    tmp8 = tmp0 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, float("-inf"))
    tmp14 = triton_helpers.max2(tmp13, 1)[:, None]
    tmp15 = tmp10 - tmp14
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp16 / tmp20
    tl.store(out_ptr3 + (r3 + (49*x0) + (2432*x4)), tmp21, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ua/cua2bthrv3lqedvcklbce2nut5lxkrfnmoyf57qpxu5ifep2d63a.py
# Topologically Sorted Source Nodes: [layer_norm_57], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_57 => add_276, add_277, mul_215, mul_216, rsqrt_57, sub_83, var_mean_57
# Graph fragment:
#   %var_mean_57 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_706, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_706, %getitem_193), kwargs = {})
#   %add_276 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_192, 1e-05), kwargs = {})
#   %rsqrt_57 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_276,), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %rsqrt_57), kwargs = {})
#   %mul_216 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_215, %arg28_1), kwargs = {})
#   %add_277 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_216, %arg29_1), kwargs = {})
triton_red_fused_native_layer_norm_13 = async_compile.triton('triton_red_fused_native_layer_norm_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*(((53 + (x0 % 56)) % 56) % 7)) + (896*(((53 + (x0 // 56)) % 56) % 7)) + (6272*(((53 + (x0 % 56)) % 56) // 7)) + (50176*(triton_helpers.div_floor_integer((53 + (x0 // 56)) % 56,  7))) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
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
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r2 + (128*(((53 + (x0 % 56)) % 56) % 7)) + (896*(((53 + (x0 // 56)) % 56) % 7)) + (6272*(((53 + (x0 % 56)) % 56) // 7)) + (50176*(triton_helpers.div_floor_integer((53 + (x0 // 56)) % 56,  7))) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp9 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 128.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-05
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2d/c2dcfeplfrjexnqyhkzeqqexphcuwwpvmszvfxfskhq74xwsdvwp.py
# Topologically Sorted Source Nodes: [x_451], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_451 => clone_288
# Graph fragment:
#   %clone_288 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_269,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_14 = async_compile.triton('triton_poi_fused_clone_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 56
    x2 = (xindex // 7168) % 56
    x3 = (xindex // 401408)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*(((53 + x1) % 56) % 7)) + (896*(((53 + x2) % 56) % 7)) + (6272*(((53 + x1) % 56) // 7)) + (50176*(((53 + x2) % 56) // 7)) + (401408*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x4), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x4), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gl/cglmtrvpofmxjxx4u6wsnqemi7srrhxsfvkwk6656qmepbsthgzi.py
# Topologically Sorted Source Nodes: [x_452], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_452 => add_280, add_281, mul_220, mul_221, rsqrt_58, sub_84, var_mean_58
# Graph fragment:
#   %var_mean_58 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_713, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_713, %getitem_195), kwargs = {})
#   %add_280 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_194, 1e-05), kwargs = {})
#   %rsqrt_58 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_280,), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %rsqrt_58), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_220, %arg34_1), kwargs = {})
#   %add_281 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_221, %arg35_1), kwargs = {})
triton_per_fused_native_layer_norm_15 = async_compile.triton('triton_per_fused_native_layer_norm_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_15(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
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
    r2 = rindex
    x0 = xindex % 28
    x1 = (xindex // 28)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*(r2 // 256)) + (256*x0) + (7168*((r2 // 128) % 2)) + (14336*x1) + (r2 % 128)), None)
    tmp21 = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 512, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5g/c5gzeapyf7m5upmea236t4l466qyrrdnxv7t5zj2kjnhkkygmgx5.py
# Topologically Sorted Source Nodes: [shifted_x_78, contiguous_104], Original ATen: [aten.native_layer_norm, aten.clone]
# Source node to ATen node mapping:
#   contiguous_104 => clone_289
#   shifted_x_78 => var_mean_59
# Graph fragment:
#   %var_mean_59 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_715, [3]), kwargs = {correction: 0, keepdim: True})
#   %clone_289 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_271,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused_clone_native_layer_norm_16 = async_compile.triton('triton_per_fused_clone_native_layer_norm_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_layer_norm_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_clone_native_layer_norm_16(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
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
    x2 = xindex % 7
    x3 = (xindex // 7) % 4
    x4 = (xindex // 28) % 7
    x5 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None)
    tmp21 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 256.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr2 + (r1 + (256*x2) + (1792*x4) + (12544*x3) + (50176*x5)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fd/cfdzfnybb7qdv5aiuephnfaoc3g5764zsbdzvd57w2q2qyj6itp2.py
# Topologically Sorted Source Nodes: [q_53, attn_128], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   attn_128 => clone_290
#   q_53 => mul_224
# Graph fragment:
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_198, 0.1767766952966369), kwargs = {})
#   %clone_290 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_104,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_17 = async_compile.triton('triton_poi_fused_clone_mul_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_17(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 8
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (768*x1) + (37632*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ox/coxbw4eie7nhzw7idpaugp2tt5zzdx7gk3lp53vbrqhtu3hpsosj.py
# Topologically Sorted Source Nodes: [attn_128], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_128 => clone_291
# Graph fragment:
#   %clone_291 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_105,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_18 = async_compile.triton('triton_poi_fused_clone_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_18(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (256 + y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (256 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/k7/ck7uadrzvgll2lqm3syazjt5esfar5anzssrzvwdy3zpc3wqnxeq.py
# Topologically Sorted Source Nodes: [attn_129, attn_130], Original ATen: [aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_129 => add_284
#   attn_130 => amax_26, div_26, exp_26, sub_86, sum_27
# Graph fragment:
#   %add_284 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_724, %unsqueeze_50), kwargs = {})
#   %amax_26 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_284, [-1], True), kwargs = {})
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_284, %amax_26), kwargs = {})
#   %exp_26 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_86,), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_26, [-1], True), kwargs = {})
#   %div_26 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_26, %sum_27), kwargs = {})
triton_per_fused__softmax_add_19 = async_compile.triton('triton_per_fused__softmax_add_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_19(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 8
    x5 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.full([XBLOCK, RBLOCK], 169, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 169)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp5 < 169")
    tmp7 = tl.load(in_ptr2 + (x1 + (8*tmp5)), rmask & xmask, eviction_policy='evict_last')
    tmp8 = tmp0 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr2 + (r3 + (49*x0) + (2432*x5)), tmp19, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nh/cnhshd7ftxtdprb4vpfncrpp6ssaldld63gtw7z6rqjybjafsw7r.py
# Topologically Sorted Source Nodes: [x_455], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_455 => clone_294
# Graph fragment:
#   %clone_294 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_107,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_20 = async_compile.triton('triton_poi_fused_clone_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_20(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 8
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (32*x2) + (768*x1) + (37632*x3)), None)
    tmp1 = tl.load(in_ptr1 + (512 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sd/csd6lz4utzl23ia4e3t7jos4qyq4hrapa5yd47ec42yw2qjbhktl.py
# Topologically Sorted Source Nodes: [x_456], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_456 => clone_295
# Graph fragment:
#   %clone_295 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_276,), kwargs = {memory_format: torch.contiguous_format})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 8
    x2 = (xindex // 256) % 49
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1568*x1) + (12544*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ez/cezytzfl3mx5zlaagl7ezdzuhh35enncm4v4slhkofon5pyw4qnr.py
# Topologically Sorted Source Nodes: [layer_norm_60], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_60 => add_286, add_287, mul_225, mul_226, rsqrt_60, sub_87, var_mean_60
# Graph fragment:
#   %var_mean_60 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_736, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_736, %getitem_202), kwargs = {})
#   %add_286 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_201, 1e-05), kwargs = {})
#   %rsqrt_60 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_286,), kwargs = {})
#   %mul_225 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %rsqrt_60), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_225, %arg45_1), kwargs = {})
#   %add_287 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_226, %arg46_1), kwargs = {})
triton_per_fused_native_layer_norm_22 = async_compile.triton('triton_per_fused_native_layer_norm_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*((x0 % 28) % 7)) + (1792*((x0 // 28) % 7)) + (12544*((x0 % 28) // 7)) + (50176*(x0 // 196)) + (200704*x1)), None)
    tmp2 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
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
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kg/ckgnvhb7rwswbdy6ptvamy3h7swpkh6rli2erljgpks2o5gqq5gs.py
# Topologically Sorted Source Nodes: [x_464], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_464 => add_288, erf_26, mul_227, mul_228, mul_229
# Graph fragment:
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_738, 0.5), kwargs = {})
#   %mul_228 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_738, 0.7071067811865476), kwargs = {})
#   %erf_26 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_228,), kwargs = {})
#   %add_288 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_26, 1), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_227, %add_288), kwargs = {})
triton_poi_fused_gelu_23 = async_compile.triton('triton_poi_fused_gelu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_23(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ka/cka67rr5xnsawfs75lkbpqpehia6zjdsrezvc2phfu6nyjsczxd2.py
# Topologically Sorted Source Nodes: [x_468, layer_norm_61], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_61 => var_mean_61
#   x_468 => add_289
# Graph fragment:
#   %add_289 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_736, %view_740), kwargs = {})
#   %var_mean_61 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_741, [3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_24 = async_compile.triton('triton_per_fused_add_native_layer_norm_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*((x0 % 28) % 7)) + (1792*((x0 // 28) % 7)) + (12544*((x0 % 28) // 7)) + (50176*(x0 // 196)) + (200704*x1)), None)
    tmp2 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r2 + (256*x3)), None)
    tmp6 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp8, None)
    tl.store(out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr1 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7o/c7oauy5igth4veroujv65ex2il3dpihx4s7cdyq363xummr6nwrz.py
# Topologically Sorted Source Nodes: [contiguous_108], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_108 => clone_300
# Graph fragment:
#   %clone_300 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_281,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_25 = async_compile.triton('triton_poi_fused_clone_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x5 = (xindex // 200704)
    x6 = (xindex // 256) % 28
    x7 = (xindex // 7168) % 28
    x2 = (xindex // 1792) % 4
    x3 = (xindex // 7168) % 7
    x8 = xindex % 1792
    x9 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x0 + (256*((3 + x6) % 28)) + (7168*((3 + x7) % 28)) + (200704*x5)), None)
    tmp1 = tl.load(in_ptr1 + ((28*((3 + x7) % 28)) + (784*x5) + ((3 + x6) % 28)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((28*((3 + x7) % 28)) + (784*x5) + ((3 + x6) % 28)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 256.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x8 + (1792*x3) + (12544*x2) + (50176*x9)), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xi/cxixi3lnfqop2phgucshfippdeklq4fp4gtpmqxgm2vczhorpavm.py
# Topologically Sorted Source Nodes: [attn_136], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_136 => amax_27, div_27, exp_27, sub_89, sum_28
# Graph fragment:
#   %amax_27 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_754, [-1], True), kwargs = {})
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_754, %amax_27), kwargs = {})
#   %exp_27 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_89,), kwargs = {})
#   %sum_28 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_27, [-1], True), kwargs = {})
#   %div_27 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_27, %sum_28), kwargs = {})
triton_per_fused__softmax_26 = async_compile.triton('triton_per_fused__softmax_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x5 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 8
    x2 = (xindex // 392)
    x4 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x5)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r3 + (49*x0) + (2401*(x2 % 16))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.full([XBLOCK, RBLOCK], 169, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 169)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp5 < 169")
    tmp7 = tl.load(in_ptr2 + (x1 + (8*tmp5)), rmask & xmask, eviction_policy='evict_last')
    tmp8 = tmp0 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, float("-inf"))
    tmp14 = triton_helpers.max2(tmp13, 1)[:, None]
    tmp15 = tmp10 - tmp14
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp16 / tmp20
    tl.store(out_ptr3 + (r3 + (49*x0) + (2432*x4)), tmp21, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fl/cfl2a6gdrz6hztshutfelsny23e6ztjmrvpnv62zsb2rbicfq6pk.py
# Topologically Sorted Source Nodes: [layer_norm_62], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_62 => add_299, add_300, mul_233, mul_234, rsqrt_62, sub_90, var_mean_62
# Graph fragment:
#   %var_mean_62 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_764, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_764, %getitem_209), kwargs = {})
#   %add_299 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_208, 1e-05), kwargs = {})
#   %rsqrt_62 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_299,), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %rsqrt_62), kwargs = {})
#   %mul_234 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_233, %arg60_1), kwargs = {})
#   %add_300 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_234, %arg61_1), kwargs = {})
triton_per_fused_native_layer_norm_27 = async_compile.triton('triton_per_fused_native_layer_norm_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*(((25 + (x0 % 28)) % 28) % 7)) + (1792*(((25 + (x0 // 28)) % 28) % 7)) + (12544*(((25 + (x0 % 28)) % 28) // 7)) + (50176*(triton_helpers.div_floor_integer((25 + (x0 // 28)) % 28,  7))) + (200704*x1)), None)
    tmp2 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
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
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rh/crhrufdohuq6agnbjpfnt7sdbf2bgyxzrbiktlizykjb7kl6l6z3.py
# Topologically Sorted Source Nodes: [x_488], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_488 => clone_311
# Graph fragment:
#   %clone_311 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_291,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_28 = async_compile.triton('triton_poi_fused_clone_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256) % 28
    x2 = (xindex // 7168) % 28
    x3 = (xindex // 200704)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*(((25 + x1) % 28) % 7)) + (1792*(((25 + x2) % 28) % 7)) + (12544*(((25 + x1) % 28) // 7)) + (50176*(((25 + x2) % 28) // 7)) + (200704*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x4), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x4), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vd/cvdvdfs2oxwnm4n57n7t5kcguaambcsaqumfde77c3plx5l4ej63.py
# Topologically Sorted Source Nodes: [x_489], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_489 => add_303, add_304, mul_238, mul_239, rsqrt_63, sub_91, var_mean_63
# Graph fragment:
#   %var_mean_63 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_771, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_771, %getitem_211), kwargs = {})
#   %add_303 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_210, 1e-05), kwargs = {})
#   %rsqrt_63 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_303,), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %rsqrt_63), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_238, %arg66_1), kwargs = {})
#   %add_304 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_239, %arg67_1), kwargs = {})
triton_per_fused_native_layer_norm_29 = async_compile.triton('triton_per_fused_native_layer_norm_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_29(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((256*(r2 // 512)) + (512*x0) + (7168*((r2 // 256) % 2)) + (14336*x1) + (r2 % 256)), None)
    tmp21 = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 1024.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3p/c3pmmvwrtqbszhnsxga4vpg3nnrlx634wvwp7zzahmkbv7xbd2yy.py
# Topologically Sorted Source Nodes: [shifted_x_84, contiguous_112], Original ATen: [aten.native_layer_norm, aten.clone]
# Source node to ATen node mapping:
#   contiguous_112 => clone_312
#   shifted_x_84 => var_mean_64
# Graph fragment:
#   %var_mean_64 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_773, [3]), kwargs = {correction: 0, keepdim: True})
#   %clone_312 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_293,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused_clone_native_layer_norm_30 = async_compile.triton('triton_per_fused_clone_native_layer_norm_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_layer_norm_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_clone_native_layer_norm_30(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
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
    x2 = xindex % 7
    x3 = (xindex // 7) % 2
    x4 = (xindex // 14) % 7
    x5 = (xindex // 98)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp21 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 512, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr2 + (r1 + (512*x2) + (3584*x4) + (25088*x3) + (50176*x5)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wu/cwuqg7eie6hnu62wkmieg24d6xr5pmxzfthkdvgyqvgganfiej3d.py
# Topologically Sorted Source Nodes: [q_57, attn_138], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   attn_138 => clone_313
#   q_57 => mul_242
# Graph fragment:
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_214, 0.1767766952966369), kwargs = {})
#   %clone_313 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_112,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_31 = async_compile.triton('triton_poi_fused_clone_mul_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_31(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 16
    x3 = (xindex // 25088)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1536*x1) + (75264*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ar/car2jvwazaukoq7cxbnrfktxlrurby363lknq6fiy5vkab7qqktf.py
# Topologically Sorted Source Nodes: [attn_138], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_138 => clone_314
# Graph fragment:
#   %clone_314 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_113,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_32 = async_compile.triton('triton_poi_fused_clone_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_32(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (512 + y0 + (1536*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (512 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/72/c72peqfgobtwji2ktbe5v5ggywvlnzejekbcfchmezoht6m6v57p.py
# Topologically Sorted Source Nodes: [attn_139, attn_140], Original ATen: [aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_139 => add_307
#   attn_140 => amax_28, div_28, exp_28, sub_93, sum_29
# Graph fragment:
#   %add_307 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_782, %unsqueeze_54), kwargs = {})
#   %amax_28 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_307, [-1], True), kwargs = {})
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_307, %amax_28), kwargs = {})
#   %exp_28 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_93,), kwargs = {})
#   %sum_29 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_28, [-1], True), kwargs = {})
#   %div_28 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_28, %sum_29), kwargs = {})
triton_per_fused__softmax_add_33 = async_compile.triton('triton_per_fused__softmax_add_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_33(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 16
    x5 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.full([XBLOCK, RBLOCK], 169, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 169)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp5 < 169")
    tmp7 = tl.load(in_ptr2 + (x1 + (16*tmp5)), rmask & xmask, eviction_policy='evict_last')
    tmp8 = tmp0 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr2 + (r3 + (49*x0) + (2432*x5)), tmp19, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/k7/ck7mekksquhygysmhlzcuialsqtucubyzvexnim2kinyskpvv7xa.py
# Topologically Sorted Source Nodes: [x_492], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_492 => clone_317
# Graph fragment:
#   %clone_317 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_115,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_34 = async_compile.triton('triton_poi_fused_clone_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_34(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 16
    x3 = (xindex // 25088)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (32*x2) + (1536*x1) + (75264*x3)), None)
    tmp1 = tl.load(in_ptr1 + (1024 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qw/cqwbqvn3ulczbmo23eejftgogofjjtbzd6tbgg2et36aeksg753i.py
# Topologically Sorted Source Nodes: [x_493], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_493 => clone_318
# Graph fragment:
#   %clone_318 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_298,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_35 = async_compile.triton('triton_poi_fused_clone_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 16
    x2 = (xindex // 512) % 49
    x3 = (xindex // 25088)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1568*x1) + (25088*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cc/cccfjje57ebggyl76sgex5b35y6yx654ev56npebcgwrtsye47ak.py
# Topologically Sorted Source Nodes: [layer_norm_65], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_65 => add_309, add_310, mul_243, mul_244, rsqrt_65, sub_94, var_mean_65
# Graph fragment:
#   %var_mean_65 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_794, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_794, %getitem_218), kwargs = {})
#   %add_309 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_217, 1e-05), kwargs = {})
#   %rsqrt_65 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_309,), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %rsqrt_65), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_243, %arg77_1), kwargs = {})
#   %add_310 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_244, %arg78_1), kwargs = {})
triton_per_fused_native_layer_norm_36 = async_compile.triton('triton_per_fused_native_layer_norm_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), None)
    tmp2 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
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
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gd/cgdm5pwcaln2udlgwxiw4iokhmboam7cfneal76kax5rxn64embi.py
# Topologically Sorted Source Nodes: [x_501], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_501 => add_311, erf_28, mul_245, mul_246, mul_247
# Graph fragment:
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_796, 0.5), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_796, 0.7071067811865476), kwargs = {})
#   %erf_28 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_246,), kwargs = {})
#   %add_311 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_28, 1), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_245, %add_311), kwargs = {})
triton_poi_fused_gelu_37 = async_compile.triton('triton_poi_fused_gelu_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_37(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/nl/cnlvct2trwftf2rzh7s27s6ml32atjenqfzs6h5nxnfkqdklcmd3.py
# Topologically Sorted Source Nodes: [x_505, layer_norm_66], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_66 => var_mean_66
#   x_505 => add_312
# Graph fragment:
#   %add_312 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_794, %view_798), kwargs = {})
#   %var_mean_66 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_799, [3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_38 = async_compile.triton('triton_per_fused_add_native_layer_norm_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), None)
    tmp2 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r2 + (512*x3)), None)
    tmp6 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp8, None)
    tl.store(out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr1 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/z2/cz2m7grdbny5w6h5nyt2kckz4u2wfx3xagxreildmf4wgwgz3g6c.py
# Topologically Sorted Source Nodes: [contiguous_116], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_116 => clone_323
# Graph fragment:
#   %clone_323 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_303,), kwargs = {memory_format: torch.contiguous_format})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x5 = (xindex // 100352)
    x6 = (xindex // 512) % 14
    x7 = (xindex // 7168) % 14
    x2 = (xindex // 3584) % 2
    x3 = (xindex // 7168) % 7
    x8 = xindex % 3584
    x9 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((3 + x6) % 14)) + (7168*((3 + x7) % 14)) + (100352*x5)), None)
    tmp1 = tl.load(in_ptr1 + ((14*((3 + x7) % 14)) + (196*x5) + ((3 + x6) % 14)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((14*((3 + x7) % 14)) + (196*x5) + ((3 + x6) % 14)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x8 + (3584*x3) + (25088*x2) + (50176*x9)), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ql/cqlqvq44pzl5ajta47bnyfzqub5ns53va5vgagkabopjxwpnfdru.py
# Topologically Sorted Source Nodes: [attn_146], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_146 => amax_29, div_29, exp_29, sub_96, sum_30
# Graph fragment:
#   %amax_29 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_812, [-1], True), kwargs = {})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_812, %amax_29), kwargs = {})
#   %exp_29 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_96,), kwargs = {})
#   %sum_30 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_29, [-1], True), kwargs = {})
#   %div_29 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_29, %sum_30), kwargs = {})
triton_per_fused__softmax_40 = async_compile.triton('triton_per_fused__softmax_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x5 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 16
    x2 = (xindex // 784)
    x4 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x5)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r3 + (49*x0) + (2401*(x2 % 4))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.full([XBLOCK, RBLOCK], 169, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 169)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp5 < 169")
    tmp7 = tl.load(in_ptr2 + (x1 + (16*tmp5)), rmask & xmask, eviction_policy='evict_last')
    tmp8 = tmp0 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, float("-inf"))
    tmp14 = triton_helpers.max2(tmp13, 1)[:, None]
    tmp15 = tmp10 - tmp14
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp16 / tmp20
    tl.store(out_ptr3 + (r3 + (49*x0) + (2432*x4)), tmp21, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/da/cdaxuertfq2nip2z7fnydlul6v6dndicw4inn2nqwdnk7ttcnwkp.py
# Topologically Sorted Source Nodes: [layer_norm_67], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_67 => add_322, add_323, mul_251, mul_252, rsqrt_67, sub_97, var_mean_67
# Graph fragment:
#   %var_mean_67 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_822, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_97 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_822, %getitem_225), kwargs = {})
#   %add_322 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_224, 1e-05), kwargs = {})
#   %rsqrt_67 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_322,), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_97, %rsqrt_67), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_251, %arg92_1), kwargs = {})
#   %add_323 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_252, %arg93_1), kwargs = {})
triton_per_fused_native_layer_norm_41 = async_compile.triton('triton_per_fused_native_layer_norm_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(triton_helpers.div_floor_integer((11 + (x0 // 14)) % 14,  7))) + (100352*x1)), None)
    tmp2 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
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
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/45/c45t4soztx6p6nm7rgpehxalgtteafcljxxswjsmtrqknyfshwkg.py
# Topologically Sorted Source Nodes: [x_522, shifted_x_90, contiguous_120], Original ATen: [aten.add, aten.native_layer_norm, aten.clone]
# Source node to ATen node mapping:
#   contiguous_120 => clone_334
#   shifted_x_90 => var_mean_68
#   x_522 => add_325
# Graph fragment:
#   %add_325 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_822, %view_826), kwargs = {})
#   %var_mean_68 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_827, [3]), kwargs = {correction: 0, keepdim: True})
#   %clone_334 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_313,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused_add_clone_native_layer_norm_42 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_42', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x4 = xindex % 7
    x5 = (xindex // 7) % 2
    x6 = (xindex // 14) % 7
    x7 = (xindex // 98)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), None)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(triton_helpers.div_floor_integer((11 + (x0 // 14)) % 14,  7))) + (100352*x1)), None)
    tmp2 = tl.load(in_ptr2 + (r2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (r2 + (512*x3)), None)
    tmp6 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
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
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp8, None)
    tl.store(out_ptr2 + (r2 + (512*x4) + (3584*x6) + (25088*x5) + (50176*x7)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ot/cotggwldhsajri6qh5n33dyamg2psjppy7xdl5xeh3uzo6lmi6fn.py
# Topologically Sorted Source Nodes: [x_789], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_789 => clone_510
# Graph fragment:
#   %clone_510 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_473,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_43 = async_compile.triton('triton_poi_fused_clone_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512) % 14
    x2 = (xindex // 7168) % 14
    x3 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*(((11 + x1) % 14) % 7)) + (3584*(((11 + x2) % 14) % 7)) + (25088*(((11 + x1) % 14) // 7)) + (50176*(((11 + x2) % 14) // 7)) + (100352*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x4), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x4), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/db/cdbsa2rsrmv3t3jvxoc7swlzmkpsuejdmjxnkgm7rclmdvypytyw.py
# Topologically Sorted Source Nodes: [x_790], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_790 => add_494, add_495, mul_384, mul_385, rsqrt_100, sub_146, var_mean_100
# Graph fragment:
#   %var_mean_100 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1261, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_146 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1261, %getitem_339), kwargs = {})
#   %add_494 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_338, 1e-05), kwargs = {})
#   %rsqrt_100 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_494,), kwargs = {})
#   %mul_384 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_146, %rsqrt_100), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_384, %arg330_1), kwargs = {})
#   %add_495 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_385, %arg331_1), kwargs = {})
triton_red_fused_native_layer_norm_44 = async_compile.triton('triton_red_fused_native_layer_norm_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_44(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((512*(r2 // 1024)) + (1024*x0) + (7168*((r2 // 512) % 2)) + (14336*x1) + (r2 % 512)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr0 + ((512*(r2 // 1024)) + (1024*x0) + (7168*((r2 // 512) % 2)) + (14336*x1) + (r2 % 512)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 2048.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp16, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gz/cgzwkod2uqb66qdtpqtgtiumwddewghwipr3cfjy25y3xexbjj37.py
# Topologically Sorted Source Nodes: [shifted_x_138], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   shifted_x_138 => add_496, add_497, mul_386, mul_387, rsqrt_101, sub_147, var_mean_101
# Graph fragment:
#   %var_mean_101 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1263, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_147 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1263, %getitem_341), kwargs = {})
#   %add_496 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_340, 1e-05), kwargs = {})
#   %rsqrt_101 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_496,), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_147, %rsqrt_101), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_386, %arg333_1), kwargs = {})
#   %add_497 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_387, %arg334_1), kwargs = {})
triton_per_fused_native_layer_norm_45 = async_compile.triton('triton_per_fused_native_layer_norm_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_45(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 392
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
    tmp21 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp0 - tmp8
    tmp15 = 1024.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/y7/cy7nny3m6otnv5kwok6whhyz2xrmayybsq4fhi76rqbcgnoabdld.py
# Topologically Sorted Source Nodes: [q_93, attn_228], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   attn_228 => clone_511
#   q_93 => mul_388
# Graph fragment:
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_342, 0.1767766952966369), kwargs = {})
#   %clone_511 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_184,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_46 = async_compile.triton('triton_poi_fused_clone_mul_46', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_46(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 32
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (3072*x1) + (150528*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ly/clythd5focimhyhmhicfxehp3am22dtm5amdj6dgvopvsiz65h3r.py
# Topologically Sorted Source Nodes: [attn_228], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_228 => clone_512
# Graph fragment:
#   %clone_512 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_185,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_47 = async_compile.triton('triton_poi_fused_clone_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_47(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (1024 + y0 + (3072*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1024 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3q/c3qjqbibi6jguofjxujol47rgweluz4acgnzc2fv5nbybriahs2z.py
# Topologically Sorted Source Nodes: [attn_229, attn_230], Original ATen: [aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_229 => add_498
#   attn_230 => amax_46, div_46, exp_46, sub_148, sum_47
# Graph fragment:
#   %add_498 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1272, %unsqueeze_90), kwargs = {})
#   %amax_46 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_498, [-1], True), kwargs = {})
#   %sub_148 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_498, %amax_46), kwargs = {})
#   %exp_46 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_148,), kwargs = {})
#   %sum_47 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_46, [-1], True), kwargs = {})
#   %div_46 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_46, %sum_47), kwargs = {})
triton_per_fused__softmax_add_48 = async_compile.triton('triton_per_fused__softmax_add_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_48(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 32
    x5 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.full([XBLOCK, RBLOCK], 169, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 169)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp5 < 169")
    tmp7 = tl.load(in_ptr2 + (x1 + (32*tmp5)), rmask & xmask, eviction_policy='evict_last')
    tmp8 = tmp0 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr2 + (r3 + (49*x0) + (2432*x5)), tmp19, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2w/c2wond7hthgld57powqbl3uiqch6wse46e27ejureutmdmtm5k3l.py
# Topologically Sorted Source Nodes: [x_793], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_793 => clone_515
# Graph fragment:
#   %clone_515 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_187,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_49 = async_compile.triton('triton_poi_fused_clone_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_49(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 32
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + (32*x2) + (3072*x1) + (150528*x3)), None)
    tmp1 = tl.load(in_ptr1 + (2048 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tf/ctfv46ljkvljeguwqeb3cxtqibu44awpnuc3d4togc3y23poffmm.py
# Topologically Sorted Source Nodes: [x_794], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_794 => clone_516
# Graph fragment:
#   %clone_516 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_480,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_50 = async_compile.triton('triton_poi_fused_clone_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_50(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 32
    x2 = (xindex // 1024) % 49
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1568*x1) + (50176*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dw/cdwwtsq4jdd5aoybysrqh6da3cmhktou2olox32yhwhvijchmr4g.py
# Topologically Sorted Source Nodes: [layer_norm_102], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_102 => add_500, add_501, mul_389, mul_390, rsqrt_102, sub_149, var_mean_102
# Graph fragment:
#   %var_mean_102 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1284, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_149 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1284, %getitem_346), kwargs = {})
#   %add_500 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_345, 1e-05), kwargs = {})
#   %rsqrt_102 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_500,), kwargs = {})
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_149, %rsqrt_102), kwargs = {})
#   %mul_390 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_389, %arg341_1), kwargs = {})
#   %add_501 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_390, %arg342_1), kwargs = {})
triton_per_fused_native_layer_norm_51 = async_compile.triton('triton_per_fused_native_layer_norm_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 392
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
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4a/c4azqrroqbll5k3imqomf4fahigfcttrwpq4jkrm4yffewv5cwr4.py
# Topologically Sorted Source Nodes: [x_802], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_802 => add_502, erf_46, mul_391, mul_392, mul_393
# Graph fragment:
#   %mul_391 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1286, 0.5), kwargs = {})
#   %mul_392 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1286, 0.7071067811865476), kwargs = {})
#   %erf_46 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_392,), kwargs = {})
#   %add_502 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_46, 1), kwargs = {})
#   %mul_393 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_391, %add_502), kwargs = {})
triton_poi_fused_gelu_52 = async_compile.triton('triton_poi_fused_gelu_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_52(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_sahanp/4q/c4qgrldgkb35ren54crgtc5hwrscmyatz2zeju7nmnpo2kcy25sb.py
# Topologically Sorted Source Nodes: [x_806, shifted_x_141], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   shifted_x_141 => add_504, add_505, mul_394, mul_395, rsqrt_103, sub_150, var_mean_103
#   x_806 => add_503
# Graph fragment:
#   %add_503 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1284, %view_1288), kwargs = {})
#   %var_mean_103 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1289, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_150 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1289, %getitem_348), kwargs = {})
#   %add_504 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_347, 1e-05), kwargs = {})
#   %rsqrt_103 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_504,), kwargs = {})
#   %mul_394 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_150, %rsqrt_103), kwargs = {})
#   %mul_395 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_394, %arg347_1), kwargs = {})
#   %add_505 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_395, %arg348_1), kwargs = {})
triton_per_fused_add_native_layer_norm_53 = async_compile.triton('triton_per_fused_add_native_layer_norm_53', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_53(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 392
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
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, None)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bb/cbbajs67w4d7boh4vpfdkmxcvpzvwj72rhopkt3mvsdd6g2baad4.py
# Topologically Sorted Source Nodes: [x_822, x_824], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_822 => add_511
#   x_824 => var_mean_105
# Graph fragment:
#   %add_511 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1310, %view_1314), kwargs = {})
#   %var_mean_105 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1315, [3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_54 = async_compile.triton('triton_per_fused_add_native_layer_norm_54', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 392
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
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, None)
    tl.store(out_ptr0 + (x0), tmp16, None)
    tl.store(out_ptr1 + (x0), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rs/crskofupyf7upuhpxfe3bpitzi4zlbujdkfg3bxibsf6h4ozljyc.py
# Topologically Sorted Source Nodes: [x_824, x_825], Original ATen: [aten.native_layer_norm, aten.mean]
# Source node to ATen node mapping:
#   x_824 => add_512, add_513, mul_402, mul_403, rsqrt_105, sub_153, var_mean_105
#   x_825 => mean_1
# Graph fragment:
#   %var_mean_105 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_1315, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_153 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1315, %getitem_355), kwargs = {})
#   %add_512 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_354, 1e-05), kwargs = {})
#   %rsqrt_105 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_512,), kwargs = {})
#   %mul_402 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_153, %rsqrt_105), kwargs = {})
#   %mul_403 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_402, %arg361_1), kwargs = {})
#   %add_513 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_403, %arg362_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_513, [1, 2]), kwargs = {})
triton_per_fused_mean_native_layer_norm_55 = async_compile.triton('triton_per_fused_mean_native_layer_norm_55', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_native_layer_norm_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (50176*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (49*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (49*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1024.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = 49.0
    tmp19 = tmp17 / tmp18
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (384, 128), (128, 1))
    assert_size_stride(arg8_1, (384, ), (1, ))
    assert_size_stride(arg9_1, (169, 4), (4, 1))
    assert_size_stride(arg10_1, (49, 49), (49, 1))
    assert_size_stride(arg11_1, (128, 128), (128, 1))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (512, 128), (128, 1))
    assert_size_stride(arg16_1, (512, ), (1, ))
    assert_size_stride(arg17_1, (128, 512), (512, 1))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (64, 49, 49), (2401, 49, 1))
    assert_size_stride(arg22_1, (384, 128), (128, 1))
    assert_size_stride(arg23_1, (384, ), (1, ))
    assert_size_stride(arg24_1, (169, 4), (4, 1))
    assert_size_stride(arg25_1, (49, 49), (49, 1))
    assert_size_stride(arg26_1, (128, 128), (128, 1))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (512, 128), (128, 1))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (128, 512), (512, 1))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (256, 512), (512, 1))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (768, 256), (256, 1))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (169, 8), (8, 1))
    assert_size_stride(arg42_1, (49, 49), (49, 1))
    assert_size_stride(arg43_1, (256, 256), (256, 1))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (1024, 256), (256, 1))
    assert_size_stride(arg48_1, (1024, ), (1, ))
    assert_size_stride(arg49_1, (256, 1024), (1024, 1))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (16, 49, 49), (2401, 49, 1))
    assert_size_stride(arg54_1, (768, 256), (256, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (169, 8), (8, 1))
    assert_size_stride(arg57_1, (49, 49), (49, 1))
    assert_size_stride(arg58_1, (256, 256), (256, 1))
    assert_size_stride(arg59_1, (256, ), (1, ))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (1024, 256), (256, 1))
    assert_size_stride(arg63_1, (1024, ), (1, ))
    assert_size_stride(arg64_1, (256, 1024), (1024, 1))
    assert_size_stride(arg65_1, (256, ), (1, ))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (512, 1024), (1024, 1))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (1536, 512), (512, 1))
    assert_size_stride(arg72_1, (1536, ), (1, ))
    assert_size_stride(arg73_1, (169, 16), (16, 1))
    assert_size_stride(arg74_1, (49, 49), (49, 1))
    assert_size_stride(arg75_1, (512, 512), (512, 1))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (2048, 512), (512, 1))
    assert_size_stride(arg80_1, (2048, ), (1, ))
    assert_size_stride(arg81_1, (512, 2048), (2048, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg86_1, (1536, 512), (512, 1))
    assert_size_stride(arg87_1, (1536, ), (1, ))
    assert_size_stride(arg88_1, (169, 16), (16, 1))
    assert_size_stride(arg89_1, (49, 49), (49, 1))
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
    assert_size_stride(arg102_1, (169, 16), (16, 1))
    assert_size_stride(arg103_1, (49, 49), (49, 1))
    assert_size_stride(arg104_1, (512, 512), (512, 1))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (512, ), (1, ))
    assert_size_stride(arg107_1, (512, ), (1, ))
    assert_size_stride(arg108_1, (2048, 512), (512, 1))
    assert_size_stride(arg109_1, (2048, ), (1, ))
    assert_size_stride(arg110_1, (512, 2048), (2048, 1))
    assert_size_stride(arg111_1, (512, ), (1, ))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg115_1, (1536, 512), (512, 1))
    assert_size_stride(arg116_1, (1536, ), (1, ))
    assert_size_stride(arg117_1, (169, 16), (16, 1))
    assert_size_stride(arg118_1, (49, 49), (49, 1))
    assert_size_stride(arg119_1, (512, 512), (512, 1))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (512, ), (1, ))
    assert_size_stride(arg122_1, (512, ), (1, ))
    assert_size_stride(arg123_1, (2048, 512), (512, 1))
    assert_size_stride(arg124_1, (2048, ), (1, ))
    assert_size_stride(arg125_1, (512, 2048), (2048, 1))
    assert_size_stride(arg126_1, (512, ), (1, ))
    assert_size_stride(arg127_1, (512, ), (1, ))
    assert_size_stride(arg128_1, (512, ), (1, ))
    assert_size_stride(arg129_1, (1536, 512), (512, 1))
    assert_size_stride(arg130_1, (1536, ), (1, ))
    assert_size_stride(arg131_1, (169, 16), (16, 1))
    assert_size_stride(arg132_1, (49, 49), (49, 1))
    assert_size_stride(arg133_1, (512, 512), (512, 1))
    assert_size_stride(arg134_1, (512, ), (1, ))
    assert_size_stride(arg135_1, (512, ), (1, ))
    assert_size_stride(arg136_1, (512, ), (1, ))
    assert_size_stride(arg137_1, (2048, 512), (512, 1))
    assert_size_stride(arg138_1, (2048, ), (1, ))
    assert_size_stride(arg139_1, (512, 2048), (2048, 1))
    assert_size_stride(arg140_1, (512, ), (1, ))
    assert_size_stride(arg141_1, (512, ), (1, ))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg144_1, (1536, 512), (512, 1))
    assert_size_stride(arg145_1, (1536, ), (1, ))
    assert_size_stride(arg146_1, (169, 16), (16, 1))
    assert_size_stride(arg147_1, (49, 49), (49, 1))
    assert_size_stride(arg148_1, (512, 512), (512, 1))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (512, ), (1, ))
    assert_size_stride(arg151_1, (512, ), (1, ))
    assert_size_stride(arg152_1, (2048, 512), (512, 1))
    assert_size_stride(arg153_1, (2048, ), (1, ))
    assert_size_stride(arg154_1, (512, 2048), (2048, 1))
    assert_size_stride(arg155_1, (512, ), (1, ))
    assert_size_stride(arg156_1, (512, ), (1, ))
    assert_size_stride(arg157_1, (512, ), (1, ))
    assert_size_stride(arg158_1, (1536, 512), (512, 1))
    assert_size_stride(arg159_1, (1536, ), (1, ))
    assert_size_stride(arg160_1, (169, 16), (16, 1))
    assert_size_stride(arg161_1, (49, 49), (49, 1))
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
    assert_size_stride(arg172_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg173_1, (1536, 512), (512, 1))
    assert_size_stride(arg174_1, (1536, ), (1, ))
    assert_size_stride(arg175_1, (169, 16), (16, 1))
    assert_size_stride(arg176_1, (49, 49), (49, 1))
    assert_size_stride(arg177_1, (512, 512), (512, 1))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (512, ), (1, ))
    assert_size_stride(arg180_1, (512, ), (1, ))
    assert_size_stride(arg181_1, (2048, 512), (512, 1))
    assert_size_stride(arg182_1, (2048, ), (1, ))
    assert_size_stride(arg183_1, (512, 2048), (2048, 1))
    assert_size_stride(arg184_1, (512, ), (1, ))
    assert_size_stride(arg185_1, (512, ), (1, ))
    assert_size_stride(arg186_1, (512, ), (1, ))
    assert_size_stride(arg187_1, (1536, 512), (512, 1))
    assert_size_stride(arg188_1, (1536, ), (1, ))
    assert_size_stride(arg189_1, (169, 16), (16, 1))
    assert_size_stride(arg190_1, (49, 49), (49, 1))
    assert_size_stride(arg191_1, (512, 512), (512, 1))
    assert_size_stride(arg192_1, (512, ), (1, ))
    assert_size_stride(arg193_1, (512, ), (1, ))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (2048, 512), (512, 1))
    assert_size_stride(arg196_1, (2048, ), (1, ))
    assert_size_stride(arg197_1, (512, 2048), (2048, 1))
    assert_size_stride(arg198_1, (512, ), (1, ))
    assert_size_stride(arg199_1, (512, ), (1, ))
    assert_size_stride(arg200_1, (512, ), (1, ))
    assert_size_stride(arg201_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg202_1, (1536, 512), (512, 1))
    assert_size_stride(arg203_1, (1536, ), (1, ))
    assert_size_stride(arg204_1, (169, 16), (16, 1))
    assert_size_stride(arg205_1, (49, 49), (49, 1))
    assert_size_stride(arg206_1, (512, 512), (512, 1))
    assert_size_stride(arg207_1, (512, ), (1, ))
    assert_size_stride(arg208_1, (512, ), (1, ))
    assert_size_stride(arg209_1, (512, ), (1, ))
    assert_size_stride(arg210_1, (2048, 512), (512, 1))
    assert_size_stride(arg211_1, (2048, ), (1, ))
    assert_size_stride(arg212_1, (512, 2048), (2048, 1))
    assert_size_stride(arg213_1, (512, ), (1, ))
    assert_size_stride(arg214_1, (512, ), (1, ))
    assert_size_stride(arg215_1, (512, ), (1, ))
    assert_size_stride(arg216_1, (1536, 512), (512, 1))
    assert_size_stride(arg217_1, (1536, ), (1, ))
    assert_size_stride(arg218_1, (169, 16), (16, 1))
    assert_size_stride(arg219_1, (49, 49), (49, 1))
    assert_size_stride(arg220_1, (512, 512), (512, 1))
    assert_size_stride(arg221_1, (512, ), (1, ))
    assert_size_stride(arg222_1, (512, ), (1, ))
    assert_size_stride(arg223_1, (512, ), (1, ))
    assert_size_stride(arg224_1, (2048, 512), (512, 1))
    assert_size_stride(arg225_1, (2048, ), (1, ))
    assert_size_stride(arg226_1, (512, 2048), (2048, 1))
    assert_size_stride(arg227_1, (512, ), (1, ))
    assert_size_stride(arg228_1, (512, ), (1, ))
    assert_size_stride(arg229_1, (512, ), (1, ))
    assert_size_stride(arg230_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg231_1, (1536, 512), (512, 1))
    assert_size_stride(arg232_1, (1536, ), (1, ))
    assert_size_stride(arg233_1, (169, 16), (16, 1))
    assert_size_stride(arg234_1, (49, 49), (49, 1))
    assert_size_stride(arg235_1, (512, 512), (512, 1))
    assert_size_stride(arg236_1, (512, ), (1, ))
    assert_size_stride(arg237_1, (512, ), (1, ))
    assert_size_stride(arg238_1, (512, ), (1, ))
    assert_size_stride(arg239_1, (2048, 512), (512, 1))
    assert_size_stride(arg240_1, (2048, ), (1, ))
    assert_size_stride(arg241_1, (512, 2048), (2048, 1))
    assert_size_stride(arg242_1, (512, ), (1, ))
    assert_size_stride(arg243_1, (512, ), (1, ))
    assert_size_stride(arg244_1, (512, ), (1, ))
    assert_size_stride(arg245_1, (1536, 512), (512, 1))
    assert_size_stride(arg246_1, (1536, ), (1, ))
    assert_size_stride(arg247_1, (169, 16), (16, 1))
    assert_size_stride(arg248_1, (49, 49), (49, 1))
    assert_size_stride(arg249_1, (512, 512), (512, 1))
    assert_size_stride(arg250_1, (512, ), (1, ))
    assert_size_stride(arg251_1, (512, ), (1, ))
    assert_size_stride(arg252_1, (512, ), (1, ))
    assert_size_stride(arg253_1, (2048, 512), (512, 1))
    assert_size_stride(arg254_1, (2048, ), (1, ))
    assert_size_stride(arg255_1, (512, 2048), (2048, 1))
    assert_size_stride(arg256_1, (512, ), (1, ))
    assert_size_stride(arg257_1, (512, ), (1, ))
    assert_size_stride(arg258_1, (512, ), (1, ))
    assert_size_stride(arg259_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg260_1, (1536, 512), (512, 1))
    assert_size_stride(arg261_1, (1536, ), (1, ))
    assert_size_stride(arg262_1, (169, 16), (16, 1))
    assert_size_stride(arg263_1, (49, 49), (49, 1))
    assert_size_stride(arg264_1, (512, 512), (512, 1))
    assert_size_stride(arg265_1, (512, ), (1, ))
    assert_size_stride(arg266_1, (512, ), (1, ))
    assert_size_stride(arg267_1, (512, ), (1, ))
    assert_size_stride(arg268_1, (2048, 512), (512, 1))
    assert_size_stride(arg269_1, (2048, ), (1, ))
    assert_size_stride(arg270_1, (512, 2048), (2048, 1))
    assert_size_stride(arg271_1, (512, ), (1, ))
    assert_size_stride(arg272_1, (512, ), (1, ))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (1536, 512), (512, 1))
    assert_size_stride(arg275_1, (1536, ), (1, ))
    assert_size_stride(arg276_1, (169, 16), (16, 1))
    assert_size_stride(arg277_1, (49, 49), (49, 1))
    assert_size_stride(arg278_1, (512, 512), (512, 1))
    assert_size_stride(arg279_1, (512, ), (1, ))
    assert_size_stride(arg280_1, (512, ), (1, ))
    assert_size_stride(arg281_1, (512, ), (1, ))
    assert_size_stride(arg282_1, (2048, 512), (512, 1))
    assert_size_stride(arg283_1, (2048, ), (1, ))
    assert_size_stride(arg284_1, (512, 2048), (2048, 1))
    assert_size_stride(arg285_1, (512, ), (1, ))
    assert_size_stride(arg286_1, (512, ), (1, ))
    assert_size_stride(arg287_1, (512, ), (1, ))
    assert_size_stride(arg288_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg289_1, (1536, 512), (512, 1))
    assert_size_stride(arg290_1, (1536, ), (1, ))
    assert_size_stride(arg291_1, (169, 16), (16, 1))
    assert_size_stride(arg292_1, (49, 49), (49, 1))
    assert_size_stride(arg293_1, (512, 512), (512, 1))
    assert_size_stride(arg294_1, (512, ), (1, ))
    assert_size_stride(arg295_1, (512, ), (1, ))
    assert_size_stride(arg296_1, (512, ), (1, ))
    assert_size_stride(arg297_1, (2048, 512), (512, 1))
    assert_size_stride(arg298_1, (2048, ), (1, ))
    assert_size_stride(arg299_1, (512, 2048), (2048, 1))
    assert_size_stride(arg300_1, (512, ), (1, ))
    assert_size_stride(arg301_1, (512, ), (1, ))
    assert_size_stride(arg302_1, (512, ), (1, ))
    assert_size_stride(arg303_1, (1536, 512), (512, 1))
    assert_size_stride(arg304_1, (1536, ), (1, ))
    assert_size_stride(arg305_1, (169, 16), (16, 1))
    assert_size_stride(arg306_1, (49, 49), (49, 1))
    assert_size_stride(arg307_1, (512, 512), (512, 1))
    assert_size_stride(arg308_1, (512, ), (1, ))
    assert_size_stride(arg309_1, (512, ), (1, ))
    assert_size_stride(arg310_1, (512, ), (1, ))
    assert_size_stride(arg311_1, (2048, 512), (512, 1))
    assert_size_stride(arg312_1, (2048, ), (1, ))
    assert_size_stride(arg313_1, (512, 2048), (2048, 1))
    assert_size_stride(arg314_1, (512, ), (1, ))
    assert_size_stride(arg315_1, (512, ), (1, ))
    assert_size_stride(arg316_1, (512, ), (1, ))
    assert_size_stride(arg317_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg318_1, (1536, 512), (512, 1))
    assert_size_stride(arg319_1, (1536, ), (1, ))
    assert_size_stride(arg320_1, (169, 16), (16, 1))
    assert_size_stride(arg321_1, (49, 49), (49, 1))
    assert_size_stride(arg322_1, (512, 512), (512, 1))
    assert_size_stride(arg323_1, (512, ), (1, ))
    assert_size_stride(arg324_1, (512, ), (1, ))
    assert_size_stride(arg325_1, (512, ), (1, ))
    assert_size_stride(arg326_1, (2048, 512), (512, 1))
    assert_size_stride(arg327_1, (2048, ), (1, ))
    assert_size_stride(arg328_1, (512, 2048), (2048, 1))
    assert_size_stride(arg329_1, (512, ), (1, ))
    assert_size_stride(arg330_1, (2048, ), (1, ))
    assert_size_stride(arg331_1, (2048, ), (1, ))
    assert_size_stride(arg332_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg333_1, (1024, ), (1, ))
    assert_size_stride(arg334_1, (1024, ), (1, ))
    assert_size_stride(arg335_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg336_1, (3072, ), (1, ))
    assert_size_stride(arg337_1, (169, 32), (32, 1))
    assert_size_stride(arg338_1, (49, 49), (49, 1))
    assert_size_stride(arg339_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg340_1, (1024, ), (1, ))
    assert_size_stride(arg341_1, (1024, ), (1, ))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg344_1, (4096, ), (1, ))
    assert_size_stride(arg345_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (1024, ), (1, ))
    assert_size_stride(arg348_1, (1024, ), (1, ))
    assert_size_stride(arg349_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg350_1, (3072, ), (1, ))
    assert_size_stride(arg351_1, (169, 32), (32, 1))
    assert_size_stride(arg352_1, (49, 49), (49, 1))
    assert_size_stride(arg353_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg354_1, (1024, ), (1, ))
    assert_size_stride(arg355_1, (1024, ), (1, ))
    assert_size_stride(arg356_1, (1024, ), (1, ))
    assert_size_stride(arg357_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg358_1, (4096, ), (1, ))
    assert_size_stride(arg359_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (1024, ), (1, ))
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg364_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_414], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((8, 56, 56, 1), (3136, 56, 1, 25088), torch.float32)
        buf2 = empty_strided_cuda((8, 56, 56, 1), (3136, 56, 1, 25088), torch.float32)
        # Topologically Sorted Source Nodes: [x_416], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf0, arg2_1, buf1, buf2, 25088, 128, grid=grid(25088), stream=stream0)
        buf4 = empty_strided_cuda((8, 56, 56, 128), (401408, 56, 1, 3136), torch.float32)
        # Topologically Sorted Source Nodes: [x_416], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_1.run(buf0, arg2_1, buf1, buf2, arg3_1, arg4_1, buf4, 3211264, grid=grid(3211264), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        buf8 = reinterpret_tensor(buf0, (8, 8, 8, 7, 7, 128), (401408, 50176, 6272, 896, 128, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [shifted_x_72, contiguous_96], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_red_fused_clone_native_layer_norm_2.run(buf4, arg5_1, arg6_1, buf8, 25088, 128, grid=grid(25088), stream=stream0)
        del arg5_1
        del arg6_1
        buf9 = empty_strided_cuda((25088, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf8, (25088, 128), (128, 1), 0), reinterpret_tensor(arg7_1, (128, 384), (1, 128), 0), out=buf9)
        del arg7_1
        buf10 = reinterpret_tensor(buf8, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [q_49, attn_118], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf9, arg8_1, buf10, 3211264, grid=grid(3211264), stream=stream0)
        buf11 = empty_strided_cuda((512, 4, 32, 49), (6272, 1568, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_118], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf9, arg8_1, buf11, 65536, 49, grid=grid(65536, 49), stream=stream0)
        buf12 = empty_strided_cuda((2048, 49, 49), (2401, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_118], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (2048, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf11, (2048, 32, 49), (1568, 49, 1), 0), out=buf12)
        buf15 = empty_strided_cuda((512, 4, 49, 49), (9728, 2432, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_119, attn_120], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_5.run(buf12, arg10_1, arg9_1, buf15, 100352, 49, grid=grid(100352), stream=stream0)
        del arg10_1
        del arg9_1
        buf16 = reinterpret_tensor(buf11, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_418], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf9, arg8_1, buf16, 3211264, grid=grid(3211264), stream=stream0)
        del arg8_1
        buf17 = reinterpret_tensor(buf10, (2048, 49, 32), (1568, 32, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_418], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf15, (2048, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf16, (2048, 49, 32), (1568, 32, 1), 0), out=buf17)
        buf18 = reinterpret_tensor(buf16, (512, 49, 4, 32), (6272, 128, 32, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_419], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf17, buf18, 3211264, grid=grid(3211264), stream=stream0)
        buf19 = reinterpret_tensor(buf17, (25088, 128), (128, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (25088, 128), (128, 1), 0), reinterpret_tensor(arg11_1, (128, 128), (1, 128), 0), out=buf19)
        del arg11_1
        buf23 = reinterpret_tensor(buf18, (8, 3136, 128), (401408, 128, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_55], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf4, buf19, arg12_1, arg13_1, arg14_1, buf23, 25088, 128, grid=grid(25088), stream=stream0)
        del arg13_1
        del arg14_1
        buf24 = empty_strided_cuda((25088, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (25088, 128), (128, 1), 0), reinterpret_tensor(arg15_1, (128, 512), (1, 128), 0), out=buf24)
        del arg15_1
        buf25 = reinterpret_tensor(buf24, (8, 3136, 512), (1605632, 512, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_427], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_9.run(buf25, arg16_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg16_1
        buf26 = reinterpret_tensor(buf23, (25088, 128), (128, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf25, (25088, 512), (512, 1), 0), reinterpret_tensor(arg17_1, (512, 128), (1, 512), 0), out=buf26)
        del arg17_1
        buf27 = reinterpret_tensor(buf26, (8, 3136, 128), (401408, 128, 1), 0); del buf26  # reuse
        buf28 = buf2; del buf2  # reuse
        buf29 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_431, layer_norm_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf27, buf4, buf19, arg12_1, arg18_1, buf28, buf29, 25088, 128, grid=grid(25088), stream=stream0)
        del arg12_1
        del arg18_1
        buf31 = reinterpret_tensor(buf4, (8, 8, 8, 7, 7, 128), (401408, 50176, 6272, 896, 128, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [contiguous_100], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf27, buf28, buf29, arg19_1, arg20_1, buf31, 3211264, grid=grid(3211264), stream=stream0)
        del arg19_1
        del arg20_1
        del buf28
        del buf29
        buf32 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (25088, 128), (128, 1), 0), reinterpret_tensor(arg22_1, (128, 384), (1, 128), 0), out=buf32)
        del arg22_1
        buf33 = reinterpret_tensor(buf31, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [q_51, attn_122], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf32, arg23_1, buf33, 3211264, grid=grid(3211264), stream=stream0)
        buf34 = reinterpret_tensor(buf19, (512, 4, 32, 49), (6272, 1568, 49, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [attn_122], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf32, arg23_1, buf34, 65536, 49, grid=grid(65536, 49), stream=stream0)
        buf35 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [attn_122], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (2048, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf34, (2048, 32, 49), (1568, 49, 1), 0), out=buf35)
        buf39 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [attn_126], Original ATen: [aten._softmax]
        triton_per_fused__softmax_12.run(buf35, arg25_1, arg24_1, arg21_1, buf39, 100352, 49, grid=grid(100352), stream=stream0)
        del arg21_1
        del arg24_1
        del arg25_1
        del buf35
        buf40 = reinterpret_tensor(buf34, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_434], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf32, arg23_1, buf40, 3211264, grid=grid(3211264), stream=stream0)
        del arg23_1
        del buf32
        buf41 = reinterpret_tensor(buf33, (2048, 49, 32), (1568, 32, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_434], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf39, (2048, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf40, (2048, 49, 32), (1568, 32, 1), 0), out=buf41)
        del buf39
        buf42 = reinterpret_tensor(buf40, (512, 49, 4, 32), (6272, 128, 32, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_435], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf41, buf42, 3211264, grid=grid(3211264), stream=stream0)
        buf43 = reinterpret_tensor(buf41, (25088, 128), (128, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (25088, 128), (128, 1), 0), reinterpret_tensor(arg26_1, (128, 128), (1, 128), 0), out=buf43)
        del arg26_1
        buf47 = reinterpret_tensor(buf42, (8, 3136, 128), (401408, 128, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_57], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_13.run(buf27, buf43, arg27_1, arg28_1, arg29_1, buf47, 25088, 128, grid=grid(25088), stream=stream0)
        del arg28_1
        del arg29_1
        buf48 = reinterpret_tensor(buf25, (25088, 512), (512, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (25088, 128), (128, 1), 0), reinterpret_tensor(arg30_1, (128, 512), (1, 128), 0), out=buf48)
        del arg30_1
        buf49 = reinterpret_tensor(buf48, (8, 3136, 512), (1605632, 512, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_444], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_9.run(buf49, arg31_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg31_1
        buf50 = reinterpret_tensor(buf47, (25088, 128), (128, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (25088, 512), (512, 1), 0), reinterpret_tensor(arg32_1, (512, 128), (1, 512), 0), out=buf50)
        del arg32_1
        del buf49
        buf51 = reinterpret_tensor(buf50, (8, 28, 28, 2, 2, 128), (401408, 14336, 256, 128, 7168, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_451], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf51, buf27, buf43, arg27_1, arg33_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg27_1
        del arg33_1
        del buf27
        buf55 = reinterpret_tensor(buf43, (8, 28, 28, 512), (401408, 14336, 512, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_452], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf51, arg34_1, arg35_1, buf55, 6272, 512, grid=grid(6272), stream=stream0)
        del arg34_1
        del arg35_1
        del buf51
        buf56 = empty_strided_cuda((6272, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_453], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (6272, 512), (512, 1), 0), reinterpret_tensor(arg36_1, (512, 256), (1, 512), 0), out=buf56)
        del arg36_1
        buf60 = empty_strided_cuda((8, 4, 4, 7, 7, 256), (200704, 50176, 12544, 1792, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [shifted_x_78, contiguous_104], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_per_fused_clone_native_layer_norm_16.run(buf56, arg37_1, arg38_1, buf60, 6272, 256, grid=grid(6272), stream=stream0)
        del arg37_1
        del arg38_1
        buf61 = empty_strided_cuda((6272, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (6272, 256), (256, 1), 0), reinterpret_tensor(arg39_1, (256, 768), (1, 256), 0), out=buf61)
        del arg39_1
        buf62 = reinterpret_tensor(buf60, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [q_53, attn_128], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_17.run(buf61, arg40_1, buf62, 1605632, grid=grid(1605632), stream=stream0)
        buf63 = empty_strided_cuda((128, 8, 32, 49), (12544, 1568, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_128], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf61, arg40_1, buf63, 32768, 49, grid=grid(32768, 49), stream=stream0)
        buf64 = empty_strided_cuda((1024, 49, 49), (2401, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_128], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf62, (1024, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf63, (1024, 32, 49), (1568, 49, 1), 0), out=buf64)
        buf67 = empty_strided_cuda((128, 8, 49, 49), (19456, 2432, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_129, attn_130], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_19.run(buf64, arg42_1, arg41_1, buf67, 50176, 49, grid=grid(50176), stream=stream0)
        del arg41_1
        del arg42_1
        buf68 = reinterpret_tensor(buf63, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [x_455], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf61, arg40_1, buf68, 1605632, grid=grid(1605632), stream=stream0)
        del arg40_1
        buf69 = reinterpret_tensor(buf62, (1024, 49, 32), (1568, 32, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_455], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf67, (1024, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf68, (1024, 49, 32), (1568, 32, 1), 0), out=buf69)
        buf70 = reinterpret_tensor(buf68, (128, 49, 8, 32), (12544, 256, 32, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_456], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf69, buf70, 1605632, grid=grid(1605632), stream=stream0)
        buf71 = reinterpret_tensor(buf69, (6272, 256), (256, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (6272, 256), (256, 1), 0), reinterpret_tensor(arg43_1, (256, 256), (1, 256), 0), out=buf71)
        del arg43_1
        buf75 = reinterpret_tensor(buf70, (8, 784, 256), (200704, 256, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_60], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_22.run(buf56, buf71, arg44_1, arg45_1, arg46_1, buf75, 6272, 256, grid=grid(6272), stream=stream0)
        del arg45_1
        del arg46_1
        buf76 = empty_strided_cuda((6272, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (6272, 256), (256, 1), 0), reinterpret_tensor(arg47_1, (256, 1024), (1, 256), 0), out=buf76)
        del arg47_1
        buf77 = reinterpret_tensor(buf76, (8, 784, 1024), (802816, 1024, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_464], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf77, arg48_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg48_1
        buf78 = reinterpret_tensor(buf75, (6272, 256), (256, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg49_1, (1024, 256), (1, 1024), 0), out=buf78)
        del arg49_1
        buf79 = reinterpret_tensor(buf78, (8, 784, 256), (200704, 256, 1), 0); del buf78  # reuse
        buf80 = empty_strided_cuda((8, 28, 28, 1), (784, 28, 1, 6272), torch.float32)
        buf81 = empty_strided_cuda((8, 28, 28, 1), (784, 28, 1, 6272), torch.float32)
        # Topologically Sorted Source Nodes: [x_468, layer_norm_61], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_24.run(buf79, buf56, buf71, arg44_1, arg50_1, buf80, buf81, 6272, 256, grid=grid(6272), stream=stream0)
        del arg44_1
        del arg50_1
        buf83 = reinterpret_tensor(buf71, (8, 4, 4, 7, 7, 256), (200704, 50176, 12544, 1792, 256, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [contiguous_108], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf79, buf80, buf81, arg51_1, arg52_1, buf83, 1605632, grid=grid(1605632), stream=stream0)
        del arg51_1
        del arg52_1
        del buf80
        del buf81
        buf84 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf83, (6272, 256), (256, 1), 0), reinterpret_tensor(arg54_1, (256, 768), (1, 256), 0), out=buf84)
        del arg54_1
        buf85 = reinterpret_tensor(buf83, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [q_55, attn_132], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_17.run(buf84, arg55_1, buf85, 1605632, grid=grid(1605632), stream=stream0)
        buf86 = reinterpret_tensor(buf56, (128, 8, 32, 49), (12544, 1568, 49, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [attn_132], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf84, arg55_1, buf86, 32768, 49, grid=grid(32768, 49), stream=stream0)
        buf87 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [attn_132], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf85, (1024, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf86, (1024, 32, 49), (1568, 49, 1), 0), out=buf87)
        buf91 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [attn_136], Original ATen: [aten._softmax]
        triton_per_fused__softmax_26.run(buf87, arg57_1, arg56_1, arg53_1, buf91, 50176, 49, grid=grid(50176), stream=stream0)
        del arg53_1
        del arg56_1
        del arg57_1
        del buf87
        buf92 = reinterpret_tensor(buf86, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_471], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf84, arg55_1, buf92, 1605632, grid=grid(1605632), stream=stream0)
        del arg55_1
        del buf84
        buf93 = reinterpret_tensor(buf85, (1024, 49, 32), (1568, 32, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_471], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf91, (1024, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf92, (1024, 49, 32), (1568, 32, 1), 0), out=buf93)
        del buf91
        buf94 = reinterpret_tensor(buf92, (128, 49, 8, 32), (12544, 256, 32, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_472], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf93, buf94, 1605632, grid=grid(1605632), stream=stream0)
        buf95 = reinterpret_tensor(buf93, (6272, 256), (256, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (6272, 256), (256, 1), 0), reinterpret_tensor(arg58_1, (256, 256), (1, 256), 0), out=buf95)
        del arg58_1
        buf99 = reinterpret_tensor(buf94, (8, 784, 256), (200704, 256, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_62], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_27.run(buf79, buf95, arg59_1, arg60_1, arg61_1, buf99, 6272, 256, grid=grid(6272), stream=stream0)
        del arg60_1
        del arg61_1
        buf100 = reinterpret_tensor(buf77, (6272, 1024), (1024, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (6272, 256), (256, 1), 0), reinterpret_tensor(arg62_1, (256, 1024), (1, 256), 0), out=buf100)
        del arg62_1
        buf101 = reinterpret_tensor(buf100, (8, 784, 1024), (802816, 1024, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_481], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf101, arg63_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg63_1
        buf102 = reinterpret_tensor(buf99, (6272, 256), (256, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg64_1, (1024, 256), (1, 1024), 0), out=buf102)
        del arg64_1
        del buf101
        buf103 = reinterpret_tensor(buf102, (8, 14, 14, 2, 2, 256), (200704, 14336, 512, 256, 7168, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_488], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf103, buf79, buf95, arg59_1, arg65_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg59_1
        del arg65_1
        del buf79
        buf107 = reinterpret_tensor(buf95, (8, 14, 14, 1024), (200704, 14336, 1024, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [x_489], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_29.run(buf103, arg66_1, arg67_1, buf107, 1568, 1024, grid=grid(1568), stream=stream0)
        del arg66_1
        del arg67_1
        del buf103
        buf108 = empty_strided_cuda((1568, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_490], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (1568, 1024), (1024, 1), 0), reinterpret_tensor(arg68_1, (1024, 512), (1, 1024), 0), out=buf108)
        del arg68_1
        buf112 = empty_strided_cuda((8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [shifted_x_84, contiguous_112], Original ATen: [aten.native_layer_norm, aten.clone]
        triton_per_fused_clone_native_layer_norm_30.run(buf108, arg69_1, arg70_1, buf112, 1568, 512, grid=grid(1568), stream=stream0)
        del arg69_1
        del arg70_1
        buf113 = empty_strided_cuda((1568, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf112, (1568, 512), (512, 1), 0), reinterpret_tensor(arg71_1, (512, 1536), (1, 512), 0), out=buf113)
        del arg71_1
        buf114 = reinterpret_tensor(buf112, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [q_57, attn_138], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf113, arg72_1, buf114, 802816, grid=grid(802816), stream=stream0)
        buf115 = empty_strided_cuda((32, 16, 32, 49), (25088, 1568, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_138], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf113, arg72_1, buf115, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf116 = empty_strided_cuda((512, 49, 49), (2401, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_138], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf114, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf115, (512, 32, 49), (1568, 49, 1), 0), out=buf116)
        buf119 = empty_strided_cuda((32, 16, 49, 49), (38912, 2432, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_139, attn_140], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_33.run(buf116, arg74_1, arg73_1, buf119, 25088, 49, grid=grid(25088), stream=stream0)
        del arg73_1
        del arg74_1
        buf120 = reinterpret_tensor(buf115, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_492], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf113, arg72_1, buf120, 802816, grid=grid(802816), stream=stream0)
        del arg72_1
        buf121 = reinterpret_tensor(buf114, (512, 49, 32), (1568, 32, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [x_492], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf120, (512, 49, 32), (1568, 32, 1), 0), out=buf121)
        buf122 = reinterpret_tensor(buf120, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [x_493], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf121, buf122, 802816, grid=grid(802816), stream=stream0)
        buf123 = reinterpret_tensor(buf121, (1568, 512), (512, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (1568, 512), (512, 1), 0), reinterpret_tensor(arg75_1, (512, 512), (1, 512), 0), out=buf123)
        del arg75_1
        buf127 = reinterpret_tensor(buf122, (8, 196, 512), (100352, 512, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_65], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf108, buf123, arg76_1, arg77_1, arg78_1, buf127, 1568, 512, grid=grid(1568), stream=stream0)
        del arg77_1
        del arg78_1
        buf128 = reinterpret_tensor(buf55, (1568, 2048), (2048, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (1568, 512), (512, 1), 0), reinterpret_tensor(arg79_1, (512, 2048), (1, 512), 0), out=buf128)
        del arg79_1
        buf129 = reinterpret_tensor(buf128, (8, 196, 2048), (401408, 2048, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_501], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf129, arg80_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg80_1
        buf130 = reinterpret_tensor(buf127, (1568, 512), (512, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg81_1, (2048, 512), (1, 2048), 0), out=buf130)
        del arg81_1
        buf131 = reinterpret_tensor(buf130, (8, 196, 512), (100352, 512, 1), 0); del buf130  # reuse
        buf132 = empty_strided_cuda((8, 14, 14, 1), (196, 14, 1, 1568), torch.float32)
        buf133 = empty_strided_cuda((8, 14, 14, 1), (196, 14, 1, 1568), torch.float32)
        # Topologically Sorted Source Nodes: [x_505, layer_norm_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf131, buf108, buf123, arg76_1, arg82_1, buf132, buf133, 1568, 512, grid=grid(1568), stream=stream0)
        del arg76_1
        del arg82_1
        buf135 = reinterpret_tensor(buf123, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [contiguous_116], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf131, buf132, buf133, arg83_1, arg84_1, buf135, 802816, grid=grid(802816), stream=stream0)
        del arg83_1
        del arg84_1
        buf136 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (1568, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 1536), (1, 512), 0), out=buf136)
        del arg86_1
        buf137 = reinterpret_tensor(buf135, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [q_59, attn_142], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf136, arg87_1, buf137, 802816, grid=grid(802816), stream=stream0)
        buf138 = reinterpret_tensor(buf108, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [attn_142], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf136, arg87_1, buf138, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf139 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [attn_142], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf138, (512, 32, 49), (1568, 49, 1), 0), out=buf139)
        buf143 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [attn_146], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf139, arg89_1, arg88_1, arg85_1, buf143, 25088, 49, grid=grid(25088), stream=stream0)
        del arg85_1
        del arg88_1
        del arg89_1
        buf144 = reinterpret_tensor(buf138, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [x_508], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf136, arg87_1, buf144, 802816, grid=grid(802816), stream=stream0)
        del arg87_1
        buf145 = reinterpret_tensor(buf137, (512, 49, 32), (1568, 32, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [x_508], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf143, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf144, (512, 49, 32), (1568, 32, 1), 0), out=buf145)
        buf146 = reinterpret_tensor(buf144, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_509], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf145, buf146, 802816, grid=grid(802816), stream=stream0)
        buf147 = reinterpret_tensor(buf145, (1568, 512), (512, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (1568, 512), (512, 1), 0), reinterpret_tensor(arg90_1, (512, 512), (1, 512), 0), out=buf147)
        del arg90_1
        buf151 = reinterpret_tensor(buf146, (8, 196, 512), (100352, 512, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_67], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf131, buf147, arg91_1, arg92_1, arg93_1, buf151, 1568, 512, grid=grid(1568), stream=stream0)
        del arg92_1
        del arg93_1
        buf152 = reinterpret_tensor(buf129, (1568, 2048), (2048, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (1568, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 2048), (1, 512), 0), out=buf152)
        del arg94_1
        buf153 = reinterpret_tensor(buf152, (8, 196, 2048), (401408, 2048, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [x_518], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf153, arg95_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg95_1
        buf154 = reinterpret_tensor(buf151, (1568, 512), (512, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg96_1, (2048, 512), (1, 2048), 0), out=buf154)
        del arg96_1
        buf155 = reinterpret_tensor(buf154, (8, 196, 512), (100352, 512, 1), 0); del buf154  # reuse
        buf159 = empty_strided_cuda((8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_522, shifted_x_90, contiguous_120], Original ATen: [aten.add, aten.native_layer_norm, aten.clone]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf155, buf131, buf147, arg91_1, arg97_1, arg98_1, arg99_1, buf159, 1568, 512, grid=grid(1568), stream=stream0)
        del arg91_1
        del arg97_1
        del arg98_1
        del arg99_1
        buf160 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (1568, 512), (512, 1), 0), reinterpret_tensor(arg100_1, (512, 1536), (1, 512), 0), out=buf160)
        del arg100_1
        buf161 = reinterpret_tensor(buf159, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [q_61, attn_148], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf160, arg101_1, buf161, 802816, grid=grid(802816), stream=stream0)
        buf162 = reinterpret_tensor(buf147, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [attn_148], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf160, arg101_1, buf162, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf163 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [attn_148], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf161, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf162, (512, 32, 49), (1568, 49, 1), 0), out=buf163)
        buf166 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [attn_149, attn_150], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_33.run(buf163, arg103_1, arg102_1, buf166, 25088, 49, grid=grid(25088), stream=stream0)
        del arg102_1
        del arg103_1
        buf167 = reinterpret_tensor(buf162, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_525], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf160, arg101_1, buf167, 802816, grid=grid(802816), stream=stream0)
        del arg101_1
        buf168 = reinterpret_tensor(buf161, (512, 49, 32), (1568, 32, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_525], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf166, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf167, (512, 49, 32), (1568, 32, 1), 0), out=buf168)
        buf169 = reinterpret_tensor(buf167, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_526], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf168, buf169, 802816, grid=grid(802816), stream=stream0)
        buf170 = reinterpret_tensor(buf168, (1568, 512), (512, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (1568, 512), (512, 1), 0), reinterpret_tensor(arg104_1, (512, 512), (1, 512), 0), out=buf170)
        del arg104_1
        buf174 = reinterpret_tensor(buf169, (8, 196, 512), (100352, 512, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_69], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf155, buf170, arg105_1, arg106_1, arg107_1, buf174, 1568, 512, grid=grid(1568), stream=stream0)
        del arg106_1
        del arg107_1
        buf175 = reinterpret_tensor(buf153, (1568, 2048), (2048, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (1568, 512), (512, 1), 0), reinterpret_tensor(arg108_1, (512, 2048), (1, 512), 0), out=buf175)
        del arg108_1
        buf176 = reinterpret_tensor(buf175, (8, 196, 2048), (401408, 2048, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [x_534], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf176, arg109_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg109_1
        buf177 = reinterpret_tensor(buf174, (1568, 512), (512, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg110_1, (2048, 512), (1, 2048), 0), out=buf177)
        del arg110_1
        buf178 = reinterpret_tensor(buf177, (8, 196, 512), (100352, 512, 1), 0); del buf177  # reuse
        buf179 = buf133; del buf133  # reuse
        buf180 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [x_538, layer_norm_70], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf178, buf155, buf170, arg105_1, arg111_1, buf179, buf180, 1568, 512, grid=grid(1568), stream=stream0)
        del arg105_1
        del arg111_1
        buf182 = reinterpret_tensor(buf170, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [contiguous_124], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf178, buf179, buf180, arg112_1, arg113_1, buf182, 802816, grid=grid(802816), stream=stream0)
        del arg112_1
        del arg113_1
        buf183 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1568, 512), (512, 1), 0), reinterpret_tensor(arg115_1, (512, 1536), (1, 512), 0), out=buf183)
        del arg115_1
        buf184 = reinterpret_tensor(buf182, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [q_63, attn_152], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf183, arg116_1, buf184, 802816, grid=grid(802816), stream=stream0)
        buf185 = reinterpret_tensor(buf155, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [attn_152], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf183, arg116_1, buf185, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf186 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [attn_152], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf185, (512, 32, 49), (1568, 49, 1), 0), out=buf186)
        buf190 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [attn_156], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf186, arg118_1, arg117_1, arg114_1, buf190, 25088, 49, grid=grid(25088), stream=stream0)
        del arg114_1
        del arg117_1
        del arg118_1
        buf191 = reinterpret_tensor(buf185, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [x_541], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf183, arg116_1, buf191, 802816, grid=grid(802816), stream=stream0)
        del arg116_1
        buf192 = reinterpret_tensor(buf184, (512, 49, 32), (1568, 32, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [x_541], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf190, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf191, (512, 49, 32), (1568, 32, 1), 0), out=buf192)
        buf193 = reinterpret_tensor(buf191, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [x_542], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf192, buf193, 802816, grid=grid(802816), stream=stream0)
        buf194 = reinterpret_tensor(buf192, (1568, 512), (512, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf193, (1568, 512), (512, 1), 0), reinterpret_tensor(arg119_1, (512, 512), (1, 512), 0), out=buf194)
        del arg119_1
        buf198 = reinterpret_tensor(buf193, (8, 196, 512), (100352, 512, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_71], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf178, buf194, arg120_1, arg121_1, arg122_1, buf198, 1568, 512, grid=grid(1568), stream=stream0)
        del arg121_1
        del arg122_1
        buf199 = reinterpret_tensor(buf176, (1568, 2048), (2048, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (1568, 512), (512, 1), 0), reinterpret_tensor(arg123_1, (512, 2048), (1, 512), 0), out=buf199)
        del arg123_1
        buf200 = reinterpret_tensor(buf199, (8, 196, 2048), (401408, 2048, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [x_551], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf200, arg124_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg124_1
        buf201 = reinterpret_tensor(buf198, (1568, 512), (512, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg125_1, (2048, 512), (1, 2048), 0), out=buf201)
        del arg125_1
        buf202 = reinterpret_tensor(buf201, (8, 196, 512), (100352, 512, 1), 0); del buf201  # reuse
        buf206 = reinterpret_tensor(buf131, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_555, shifted_x_96, contiguous_128], Original ATen: [aten.add, aten.native_layer_norm, aten.clone]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf202, buf178, buf194, arg120_1, arg126_1, arg127_1, arg128_1, buf206, 1568, 512, grid=grid(1568), stream=stream0)
        del arg120_1
        del arg126_1
        del arg127_1
        del arg128_1
        buf207 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (1568, 512), (512, 1), 0), reinterpret_tensor(arg129_1, (512, 1536), (1, 512), 0), out=buf207)
        del arg129_1
        buf208 = reinterpret_tensor(buf206, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [q_65, attn_158], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf207, arg130_1, buf208, 802816, grid=grid(802816), stream=stream0)
        buf209 = reinterpret_tensor(buf194, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [attn_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf207, arg130_1, buf209, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf210 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [attn_158], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf208, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf209, (512, 32, 49), (1568, 49, 1), 0), out=buf210)
        buf213 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [attn_159, attn_160], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_33.run(buf210, arg132_1, arg131_1, buf213, 25088, 49, grid=grid(25088), stream=stream0)
        del arg131_1
        del arg132_1
        buf214 = reinterpret_tensor(buf209, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [x_558], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf207, arg130_1, buf214, 802816, grid=grid(802816), stream=stream0)
        del arg130_1
        buf215 = reinterpret_tensor(buf208, (512, 49, 32), (1568, 32, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [x_558], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf214, (512, 49, 32), (1568, 32, 1), 0), out=buf215)
        buf216 = reinterpret_tensor(buf214, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [x_559], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf215, buf216, 802816, grid=grid(802816), stream=stream0)
        buf217 = reinterpret_tensor(buf215, (1568, 512), (512, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf216, (1568, 512), (512, 1), 0), reinterpret_tensor(arg133_1, (512, 512), (1, 512), 0), out=buf217)
        del arg133_1
        buf221 = reinterpret_tensor(buf216, (8, 196, 512), (100352, 512, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_73], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf202, buf217, arg134_1, arg135_1, arg136_1, buf221, 1568, 512, grid=grid(1568), stream=stream0)
        del arg135_1
        del arg136_1
        buf222 = reinterpret_tensor(buf200, (1568, 2048), (2048, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (1568, 512), (512, 1), 0), reinterpret_tensor(arg137_1, (512, 2048), (1, 512), 0), out=buf222)
        del arg137_1
        buf223 = reinterpret_tensor(buf222, (8, 196, 2048), (401408, 2048, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [x_567], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf223, arg138_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg138_1
        buf224 = reinterpret_tensor(buf221, (1568, 512), (512, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg139_1, (2048, 512), (1, 2048), 0), out=buf224)
        del arg139_1
        buf225 = reinterpret_tensor(buf224, (8, 196, 512), (100352, 512, 1), 0); del buf224  # reuse
        buf226 = buf180; del buf180  # reuse
        buf227 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_571, layer_norm_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf225, buf202, buf217, arg134_1, arg140_1, buf226, buf227, 1568, 512, grid=grid(1568), stream=stream0)
        del arg134_1
        del arg140_1
        buf229 = reinterpret_tensor(buf217, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [contiguous_132], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf225, buf226, buf227, arg141_1, arg142_1, buf229, 802816, grid=grid(802816), stream=stream0)
        del arg141_1
        del arg142_1
        buf230 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (1568, 512), (512, 1), 0), reinterpret_tensor(arg144_1, (512, 1536), (1, 512), 0), out=buf230)
        del arg144_1
        buf231 = reinterpret_tensor(buf229, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [q_67, attn_162], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf230, arg145_1, buf231, 802816, grid=grid(802816), stream=stream0)
        buf232 = reinterpret_tensor(buf202, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [attn_162], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf230, arg145_1, buf232, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf233 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [attn_162], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf231, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf232, (512, 32, 49), (1568, 49, 1), 0), out=buf233)
        buf237 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [attn_166], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf233, arg147_1, arg146_1, arg143_1, buf237, 25088, 49, grid=grid(25088), stream=stream0)
        del arg143_1
        del arg146_1
        del arg147_1
        buf238 = reinterpret_tensor(buf232, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_574], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf230, arg145_1, buf238, 802816, grid=grid(802816), stream=stream0)
        del arg145_1
        buf239 = reinterpret_tensor(buf231, (512, 49, 32), (1568, 32, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [x_574], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf237, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf238, (512, 49, 32), (1568, 32, 1), 0), out=buf239)
        buf240 = reinterpret_tensor(buf238, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [x_575], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf239, buf240, 802816, grid=grid(802816), stream=stream0)
        buf241 = reinterpret_tensor(buf239, (1568, 512), (512, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (1568, 512), (512, 1), 0), reinterpret_tensor(arg148_1, (512, 512), (1, 512), 0), out=buf241)
        del arg148_1
        buf245 = reinterpret_tensor(buf240, (8, 196, 512), (100352, 512, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_75], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf225, buf241, arg149_1, arg150_1, arg151_1, buf245, 1568, 512, grid=grid(1568), stream=stream0)
        del arg150_1
        del arg151_1
        buf246 = reinterpret_tensor(buf223, (1568, 2048), (2048, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (1568, 512), (512, 1), 0), reinterpret_tensor(arg152_1, (512, 2048), (1, 512), 0), out=buf246)
        del arg152_1
        buf247 = reinterpret_tensor(buf246, (8, 196, 2048), (401408, 2048, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [x_584], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf247, arg153_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg153_1
        buf248 = reinterpret_tensor(buf245, (1568, 512), (512, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg154_1, (2048, 512), (1, 2048), 0), out=buf248)
        del arg154_1
        buf249 = reinterpret_tensor(buf248, (8, 196, 512), (100352, 512, 1), 0); del buf248  # reuse
        buf253 = reinterpret_tensor(buf178, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_588, shifted_x_102, contiguous_136], Original ATen: [aten.add, aten.native_layer_norm, aten.clone]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf249, buf225, buf241, arg149_1, arg155_1, arg156_1, arg157_1, buf253, 1568, 512, grid=grid(1568), stream=stream0)
        del arg149_1
        del arg155_1
        del arg156_1
        del arg157_1
        buf254 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (1568, 512), (512, 1), 0), reinterpret_tensor(arg158_1, (512, 1536), (1, 512), 0), out=buf254)
        del arg158_1
        buf255 = reinterpret_tensor(buf253, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [q_69, attn_168], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf254, arg159_1, buf255, 802816, grid=grid(802816), stream=stream0)
        buf256 = reinterpret_tensor(buf241, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [attn_168], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf254, arg159_1, buf256, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf257 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [attn_168], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf255, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf256, (512, 32, 49), (1568, 49, 1), 0), out=buf257)
        buf260 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [attn_169, attn_170], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_33.run(buf257, arg161_1, arg160_1, buf260, 25088, 49, grid=grid(25088), stream=stream0)
        del arg160_1
        del arg161_1
        buf261 = reinterpret_tensor(buf256, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [x_591], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf254, arg159_1, buf261, 802816, grid=grid(802816), stream=stream0)
        del arg159_1
        buf262 = reinterpret_tensor(buf255, (512, 49, 32), (1568, 32, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [x_591], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf260, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf261, (512, 49, 32), (1568, 32, 1), 0), out=buf262)
        buf263 = reinterpret_tensor(buf261, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [x_592], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf262, buf263, 802816, grid=grid(802816), stream=stream0)
        buf264 = reinterpret_tensor(buf262, (1568, 512), (512, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf263, (1568, 512), (512, 1), 0), reinterpret_tensor(arg162_1, (512, 512), (1, 512), 0), out=buf264)
        del arg162_1
        buf268 = reinterpret_tensor(buf263, (8, 196, 512), (100352, 512, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_77], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf249, buf264, arg163_1, arg164_1, arg165_1, buf268, 1568, 512, grid=grid(1568), stream=stream0)
        del arg164_1
        del arg165_1
        buf269 = reinterpret_tensor(buf247, (1568, 2048), (2048, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf268, (1568, 512), (512, 1), 0), reinterpret_tensor(arg166_1, (512, 2048), (1, 512), 0), out=buf269)
        del arg166_1
        buf270 = reinterpret_tensor(buf269, (8, 196, 2048), (401408, 2048, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [x_600], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf270, arg167_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg167_1
        buf271 = reinterpret_tensor(buf268, (1568, 512), (512, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf270, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg168_1, (2048, 512), (1, 2048), 0), out=buf271)
        del arg168_1
        buf272 = reinterpret_tensor(buf271, (8, 196, 512), (100352, 512, 1), 0); del buf271  # reuse
        buf273 = buf227; del buf227  # reuse
        buf274 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [x_604, layer_norm_78], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf272, buf249, buf264, arg163_1, arg169_1, buf273, buf274, 1568, 512, grid=grid(1568), stream=stream0)
        del arg163_1
        del arg169_1
        buf276 = reinterpret_tensor(buf264, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [contiguous_140], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf272, buf273, buf274, arg170_1, arg171_1, buf276, 802816, grid=grid(802816), stream=stream0)
        del arg170_1
        del arg171_1
        buf277 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (1568, 512), (512, 1), 0), reinterpret_tensor(arg173_1, (512, 1536), (1, 512), 0), out=buf277)
        del arg173_1
        buf278 = reinterpret_tensor(buf276, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [q_71, attn_172], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf277, arg174_1, buf278, 802816, grid=grid(802816), stream=stream0)
        buf279 = reinterpret_tensor(buf249, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [attn_172], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf277, arg174_1, buf279, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf280 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [attn_172], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf279, (512, 32, 49), (1568, 49, 1), 0), out=buf280)
        buf284 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [attn_176], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf280, arg176_1, arg175_1, arg172_1, buf284, 25088, 49, grid=grid(25088), stream=stream0)
        del arg172_1
        del arg175_1
        del arg176_1
        buf285 = reinterpret_tensor(buf279, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [x_607], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf277, arg174_1, buf285, 802816, grid=grid(802816), stream=stream0)
        del arg174_1
        buf286 = reinterpret_tensor(buf278, (512, 49, 32), (1568, 32, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [x_607], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf284, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf285, (512, 49, 32), (1568, 32, 1), 0), out=buf286)
        buf287 = reinterpret_tensor(buf285, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [x_608], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf286, buf287, 802816, grid=grid(802816), stream=stream0)
        buf288 = reinterpret_tensor(buf286, (1568, 512), (512, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf287, (1568, 512), (512, 1), 0), reinterpret_tensor(arg177_1, (512, 512), (1, 512), 0), out=buf288)
        del arg177_1
        buf292 = reinterpret_tensor(buf287, (8, 196, 512), (100352, 512, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_79], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf272, buf288, arg178_1, arg179_1, arg180_1, buf292, 1568, 512, grid=grid(1568), stream=stream0)
        del arg179_1
        del arg180_1
        buf293 = reinterpret_tensor(buf270, (1568, 2048), (2048, 1), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (1568, 512), (512, 1), 0), reinterpret_tensor(arg181_1, (512, 2048), (1, 512), 0), out=buf293)
        del arg181_1
        buf294 = reinterpret_tensor(buf293, (8, 196, 2048), (401408, 2048, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [x_617], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf294, arg182_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg182_1
        buf295 = reinterpret_tensor(buf292, (1568, 512), (512, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf294, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg183_1, (2048, 512), (1, 2048), 0), out=buf295)
        del arg183_1
        buf296 = reinterpret_tensor(buf295, (8, 196, 512), (100352, 512, 1), 0); del buf295  # reuse
        buf300 = reinterpret_tensor(buf225, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [x_621, shifted_x_108, contiguous_144], Original ATen: [aten.add, aten.native_layer_norm, aten.clone]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf296, buf272, buf288, arg178_1, arg184_1, arg185_1, arg186_1, buf300, 1568, 512, grid=grid(1568), stream=stream0)
        del arg178_1
        del arg184_1
        del arg185_1
        del arg186_1
        buf301 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf300, (1568, 512), (512, 1), 0), reinterpret_tensor(arg187_1, (512, 1536), (1, 512), 0), out=buf301)
        del arg187_1
        buf302 = reinterpret_tensor(buf300, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [q_73, attn_178], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf301, arg188_1, buf302, 802816, grid=grid(802816), stream=stream0)
        buf303 = reinterpret_tensor(buf288, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [attn_178], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf301, arg188_1, buf303, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf304 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [attn_178], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf302, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf303, (512, 32, 49), (1568, 49, 1), 0), out=buf304)
        buf307 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [attn_179, attn_180], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_33.run(buf304, arg190_1, arg189_1, buf307, 25088, 49, grid=grid(25088), stream=stream0)
        del arg189_1
        del arg190_1
        buf308 = reinterpret_tensor(buf303, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [x_624], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf301, arg188_1, buf308, 802816, grid=grid(802816), stream=stream0)
        del arg188_1
        buf309 = reinterpret_tensor(buf302, (512, 49, 32), (1568, 32, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [x_624], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf307, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf308, (512, 49, 32), (1568, 32, 1), 0), out=buf309)
        buf310 = reinterpret_tensor(buf308, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [x_625], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf309, buf310, 802816, grid=grid(802816), stream=stream0)
        buf311 = reinterpret_tensor(buf309, (1568, 512), (512, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (1568, 512), (512, 1), 0), reinterpret_tensor(arg191_1, (512, 512), (1, 512), 0), out=buf311)
        del arg191_1
        buf315 = reinterpret_tensor(buf310, (8, 196, 512), (100352, 512, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_81], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf296, buf311, arg192_1, arg193_1, arg194_1, buf315, 1568, 512, grid=grid(1568), stream=stream0)
        del arg193_1
        del arg194_1
        buf316 = reinterpret_tensor(buf294, (1568, 2048), (2048, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (1568, 512), (512, 1), 0), reinterpret_tensor(arg195_1, (512, 2048), (1, 512), 0), out=buf316)
        del arg195_1
        buf317 = reinterpret_tensor(buf316, (8, 196, 2048), (401408, 2048, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [x_633], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf317, arg196_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg196_1
        buf318 = reinterpret_tensor(buf315, (1568, 512), (512, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg197_1, (2048, 512), (1, 2048), 0), out=buf318)
        del arg197_1
        buf319 = reinterpret_tensor(buf318, (8, 196, 512), (100352, 512, 1), 0); del buf318  # reuse
        buf320 = buf274; del buf274  # reuse
        buf321 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [x_637, layer_norm_82], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf319, buf296, buf311, arg192_1, arg198_1, buf320, buf321, 1568, 512, grid=grid(1568), stream=stream0)
        del arg192_1
        del arg198_1
        buf323 = reinterpret_tensor(buf311, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [contiguous_148], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf319, buf320, buf321, arg199_1, arg200_1, buf323, 802816, grid=grid(802816), stream=stream0)
        del arg199_1
        del arg200_1
        buf324 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (1568, 512), (512, 1), 0), reinterpret_tensor(arg202_1, (512, 1536), (1, 512), 0), out=buf324)
        del arg202_1
        buf325 = reinterpret_tensor(buf323, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [q_75, attn_182], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf324, arg203_1, buf325, 802816, grid=grid(802816), stream=stream0)
        buf326 = reinterpret_tensor(buf296, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [attn_182], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf324, arg203_1, buf326, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf327 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [attn_182], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf326, (512, 32, 49), (1568, 49, 1), 0), out=buf327)
        buf331 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [attn_186], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf327, arg205_1, arg204_1, arg201_1, buf331, 25088, 49, grid=grid(25088), stream=stream0)
        del arg201_1
        del arg204_1
        del arg205_1
        buf332 = reinterpret_tensor(buf326, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [x_640], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf324, arg203_1, buf332, 802816, grid=grid(802816), stream=stream0)
        del arg203_1
        buf333 = reinterpret_tensor(buf325, (512, 49, 32), (1568, 32, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [x_640], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf331, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf332, (512, 49, 32), (1568, 32, 1), 0), out=buf333)
        buf334 = reinterpret_tensor(buf332, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [x_641], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf333, buf334, 802816, grid=grid(802816), stream=stream0)
        buf335 = reinterpret_tensor(buf333, (1568, 512), (512, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (1568, 512), (512, 1), 0), reinterpret_tensor(arg206_1, (512, 512), (1, 512), 0), out=buf335)
        del arg206_1
        buf339 = reinterpret_tensor(buf334, (8, 196, 512), (100352, 512, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_83], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf319, buf335, arg207_1, arg208_1, arg209_1, buf339, 1568, 512, grid=grid(1568), stream=stream0)
        del arg208_1
        del arg209_1
        buf340 = reinterpret_tensor(buf317, (1568, 2048), (2048, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf339, (1568, 512), (512, 1), 0), reinterpret_tensor(arg210_1, (512, 2048), (1, 512), 0), out=buf340)
        del arg210_1
        buf341 = reinterpret_tensor(buf340, (8, 196, 2048), (401408, 2048, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [x_650], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf341, arg211_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg211_1
        buf342 = reinterpret_tensor(buf339, (1568, 512), (512, 1), 0); del buf339  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf341, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg212_1, (2048, 512), (1, 2048), 0), out=buf342)
        del arg212_1
        buf343 = reinterpret_tensor(buf342, (8, 196, 512), (100352, 512, 1), 0); del buf342  # reuse
        buf347 = reinterpret_tensor(buf272, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [x_654, shifted_x_114, contiguous_152], Original ATen: [aten.add, aten.native_layer_norm, aten.clone]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf343, buf319, buf335, arg207_1, arg213_1, arg214_1, arg215_1, buf347, 1568, 512, grid=grid(1568), stream=stream0)
        del arg207_1
        del arg213_1
        del arg214_1
        del arg215_1
        buf348 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf347, (1568, 512), (512, 1), 0), reinterpret_tensor(arg216_1, (512, 1536), (1, 512), 0), out=buf348)
        del arg216_1
        buf349 = reinterpret_tensor(buf347, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [q_77, attn_188], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf348, arg217_1, buf349, 802816, grid=grid(802816), stream=stream0)
        buf350 = reinterpret_tensor(buf335, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [attn_188], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf348, arg217_1, buf350, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf351 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [attn_188], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf349, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf350, (512, 32, 49), (1568, 49, 1), 0), out=buf351)
        buf354 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [attn_189, attn_190], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_33.run(buf351, arg219_1, arg218_1, buf354, 25088, 49, grid=grid(25088), stream=stream0)
        del arg218_1
        del arg219_1
        buf355 = reinterpret_tensor(buf350, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf350  # reuse
        # Topologically Sorted Source Nodes: [x_657], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf348, arg217_1, buf355, 802816, grid=grid(802816), stream=stream0)
        del arg217_1
        buf356 = reinterpret_tensor(buf349, (512, 49, 32), (1568, 32, 1), 0); del buf349  # reuse
        # Topologically Sorted Source Nodes: [x_657], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf354, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf355, (512, 49, 32), (1568, 32, 1), 0), out=buf356)
        buf357 = reinterpret_tensor(buf355, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf355  # reuse
        # Topologically Sorted Source Nodes: [x_658], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf356, buf357, 802816, grid=grid(802816), stream=stream0)
        buf358 = reinterpret_tensor(buf356, (1568, 512), (512, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf357, (1568, 512), (512, 1), 0), reinterpret_tensor(arg220_1, (512, 512), (1, 512), 0), out=buf358)
        del arg220_1
        buf362 = reinterpret_tensor(buf357, (8, 196, 512), (100352, 512, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_85], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf343, buf358, arg221_1, arg222_1, arg223_1, buf362, 1568, 512, grid=grid(1568), stream=stream0)
        del arg222_1
        del arg223_1
        buf363 = reinterpret_tensor(buf341, (1568, 2048), (2048, 1), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf362, (1568, 512), (512, 1), 0), reinterpret_tensor(arg224_1, (512, 2048), (1, 512), 0), out=buf363)
        del arg224_1
        buf364 = reinterpret_tensor(buf363, (8, 196, 2048), (401408, 2048, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [x_666], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf364, arg225_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg225_1
        buf365 = reinterpret_tensor(buf362, (1568, 512), (512, 1), 0); del buf362  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg226_1, (2048, 512), (1, 2048), 0), out=buf365)
        del arg226_1
        buf366 = reinterpret_tensor(buf365, (8, 196, 512), (100352, 512, 1), 0); del buf365  # reuse
        buf367 = buf321; del buf321  # reuse
        buf368 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [x_670, layer_norm_86], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf366, buf343, buf358, arg221_1, arg227_1, buf367, buf368, 1568, 512, grid=grid(1568), stream=stream0)
        del arg221_1
        del arg227_1
        buf370 = reinterpret_tensor(buf358, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [contiguous_156], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf366, buf367, buf368, arg228_1, arg229_1, buf370, 802816, grid=grid(802816), stream=stream0)
        del arg228_1
        del arg229_1
        buf371 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf370, (1568, 512), (512, 1), 0), reinterpret_tensor(arg231_1, (512, 1536), (1, 512), 0), out=buf371)
        del arg231_1
        buf372 = reinterpret_tensor(buf370, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [q_79, attn_192], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf371, arg232_1, buf372, 802816, grid=grid(802816), stream=stream0)
        buf373 = reinterpret_tensor(buf343, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [attn_192], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf371, arg232_1, buf373, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf374 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [attn_192], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf372, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf373, (512, 32, 49), (1568, 49, 1), 0), out=buf374)
        buf378 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [attn_196], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf374, arg234_1, arg233_1, arg230_1, buf378, 25088, 49, grid=grid(25088), stream=stream0)
        del arg230_1
        del arg233_1
        del arg234_1
        buf379 = reinterpret_tensor(buf373, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [x_673], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf371, arg232_1, buf379, 802816, grid=grid(802816), stream=stream0)
        del arg232_1
        buf380 = reinterpret_tensor(buf372, (512, 49, 32), (1568, 32, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [x_673], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf378, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf379, (512, 49, 32), (1568, 32, 1), 0), out=buf380)
        buf381 = reinterpret_tensor(buf379, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [x_674], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf380, buf381, 802816, grid=grid(802816), stream=stream0)
        buf382 = reinterpret_tensor(buf380, (1568, 512), (512, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf381, (1568, 512), (512, 1), 0), reinterpret_tensor(arg235_1, (512, 512), (1, 512), 0), out=buf382)
        del arg235_1
        buf386 = reinterpret_tensor(buf381, (8, 196, 512), (100352, 512, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_87], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf366, buf382, arg236_1, arg237_1, arg238_1, buf386, 1568, 512, grid=grid(1568), stream=stream0)
        del arg237_1
        del arg238_1
        buf387 = reinterpret_tensor(buf364, (1568, 2048), (2048, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf386, (1568, 512), (512, 1), 0), reinterpret_tensor(arg239_1, (512, 2048), (1, 512), 0), out=buf387)
        del arg239_1
        buf388 = reinterpret_tensor(buf387, (8, 196, 2048), (401408, 2048, 1), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [x_683], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf388, arg240_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg240_1
        buf389 = reinterpret_tensor(buf386, (1568, 512), (512, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf388, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg241_1, (2048, 512), (1, 2048), 0), out=buf389)
        del arg241_1
        buf390 = reinterpret_tensor(buf389, (8, 196, 512), (100352, 512, 1), 0); del buf389  # reuse
        buf394 = reinterpret_tensor(buf319, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [x_687, shifted_x_120, contiguous_160], Original ATen: [aten.add, aten.native_layer_norm, aten.clone]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf390, buf366, buf382, arg236_1, arg242_1, arg243_1, arg244_1, buf394, 1568, 512, grid=grid(1568), stream=stream0)
        del arg236_1
        del arg242_1
        del arg243_1
        del arg244_1
        buf395 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (1568, 512), (512, 1), 0), reinterpret_tensor(arg245_1, (512, 1536), (1, 512), 0), out=buf395)
        del arg245_1
        buf396 = reinterpret_tensor(buf394, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf394  # reuse
        # Topologically Sorted Source Nodes: [q_81, attn_198], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf395, arg246_1, buf396, 802816, grid=grid(802816), stream=stream0)
        buf397 = reinterpret_tensor(buf382, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [attn_198], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf395, arg246_1, buf397, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf398 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [attn_198], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf396, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf397, (512, 32, 49), (1568, 49, 1), 0), out=buf398)
        buf401 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [attn_199, attn_200], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_33.run(buf398, arg248_1, arg247_1, buf401, 25088, 49, grid=grid(25088), stream=stream0)
        del arg247_1
        del arg248_1
        buf402 = reinterpret_tensor(buf397, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [x_690], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf395, arg246_1, buf402, 802816, grid=grid(802816), stream=stream0)
        del arg246_1
        buf403 = reinterpret_tensor(buf396, (512, 49, 32), (1568, 32, 1), 0); del buf396  # reuse
        # Topologically Sorted Source Nodes: [x_690], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf402, (512, 49, 32), (1568, 32, 1), 0), out=buf403)
        buf404 = reinterpret_tensor(buf402, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf402  # reuse
        # Topologically Sorted Source Nodes: [x_691], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf403, buf404, 802816, grid=grid(802816), stream=stream0)
        buf405 = reinterpret_tensor(buf403, (1568, 512), (512, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (1568, 512), (512, 1), 0), reinterpret_tensor(arg249_1, (512, 512), (1, 512), 0), out=buf405)
        del arg249_1
        buf409 = reinterpret_tensor(buf404, (8, 196, 512), (100352, 512, 1), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_89], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf390, buf405, arg250_1, arg251_1, arg252_1, buf409, 1568, 512, grid=grid(1568), stream=stream0)
        del arg251_1
        del arg252_1
        buf410 = reinterpret_tensor(buf388, (1568, 2048), (2048, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf409, (1568, 512), (512, 1), 0), reinterpret_tensor(arg253_1, (512, 2048), (1, 512), 0), out=buf410)
        del arg253_1
        buf411 = reinterpret_tensor(buf410, (8, 196, 2048), (401408, 2048, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [x_699], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf411, arg254_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg254_1
        buf412 = reinterpret_tensor(buf409, (1568, 512), (512, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf411, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg255_1, (2048, 512), (1, 2048), 0), out=buf412)
        del arg255_1
        buf413 = reinterpret_tensor(buf412, (8, 196, 512), (100352, 512, 1), 0); del buf412  # reuse
        buf414 = buf368; del buf368  # reuse
        buf415 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [x_703, layer_norm_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf413, buf390, buf405, arg250_1, arg256_1, buf414, buf415, 1568, 512, grid=grid(1568), stream=stream0)
        del arg250_1
        del arg256_1
        buf417 = reinterpret_tensor(buf405, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [contiguous_164], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf413, buf414, buf415, arg257_1, arg258_1, buf417, 802816, grid=grid(802816), stream=stream0)
        del arg257_1
        del arg258_1
        buf418 = buf395; del buf395  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf417, (1568, 512), (512, 1), 0), reinterpret_tensor(arg260_1, (512, 1536), (1, 512), 0), out=buf418)
        del arg260_1
        buf419 = reinterpret_tensor(buf417, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [q_83, attn_202], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf418, arg261_1, buf419, 802816, grid=grid(802816), stream=stream0)
        buf420 = reinterpret_tensor(buf390, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf390  # reuse
        # Topologically Sorted Source Nodes: [attn_202], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf418, arg261_1, buf420, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf421 = buf398; del buf398  # reuse
        # Topologically Sorted Source Nodes: [attn_202], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf419, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf420, (512, 32, 49), (1568, 49, 1), 0), out=buf421)
        buf425 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [attn_206], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf421, arg263_1, arg262_1, arg259_1, buf425, 25088, 49, grid=grid(25088), stream=stream0)
        del arg259_1
        del arg262_1
        del arg263_1
        buf426 = reinterpret_tensor(buf420, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf420  # reuse
        # Topologically Sorted Source Nodes: [x_706], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf418, arg261_1, buf426, 802816, grid=grid(802816), stream=stream0)
        del arg261_1
        buf427 = reinterpret_tensor(buf419, (512, 49, 32), (1568, 32, 1), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [x_706], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf425, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf426, (512, 49, 32), (1568, 32, 1), 0), out=buf427)
        buf428 = reinterpret_tensor(buf426, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [x_707], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf427, buf428, 802816, grid=grid(802816), stream=stream0)
        buf429 = reinterpret_tensor(buf427, (1568, 512), (512, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (1568, 512), (512, 1), 0), reinterpret_tensor(arg264_1, (512, 512), (1, 512), 0), out=buf429)
        del arg264_1
        buf433 = reinterpret_tensor(buf428, (8, 196, 512), (100352, 512, 1), 0); del buf428  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_91], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf413, buf429, arg265_1, arg266_1, arg267_1, buf433, 1568, 512, grid=grid(1568), stream=stream0)
        del arg266_1
        del arg267_1
        buf434 = reinterpret_tensor(buf411, (1568, 2048), (2048, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf433, (1568, 512), (512, 1), 0), reinterpret_tensor(arg268_1, (512, 2048), (1, 512), 0), out=buf434)
        del arg268_1
        buf435 = reinterpret_tensor(buf434, (8, 196, 2048), (401408, 2048, 1), 0); del buf434  # reuse
        # Topologically Sorted Source Nodes: [x_716], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf435, arg269_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg269_1
        buf436 = reinterpret_tensor(buf433, (1568, 512), (512, 1), 0); del buf433  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg270_1, (2048, 512), (1, 2048), 0), out=buf436)
        del arg270_1
        buf437 = reinterpret_tensor(buf436, (8, 196, 512), (100352, 512, 1), 0); del buf436  # reuse
        buf441 = reinterpret_tensor(buf366, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [x_720, shifted_x_126, contiguous_168], Original ATen: [aten.add, aten.native_layer_norm, aten.clone]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf437, buf413, buf429, arg265_1, arg271_1, arg272_1, arg273_1, buf441, 1568, 512, grid=grid(1568), stream=stream0)
        del arg265_1
        del arg271_1
        del arg272_1
        del arg273_1
        buf442 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf441, (1568, 512), (512, 1), 0), reinterpret_tensor(arg274_1, (512, 1536), (1, 512), 0), out=buf442)
        del arg274_1
        buf443 = reinterpret_tensor(buf441, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [q_85, attn_208], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf442, arg275_1, buf443, 802816, grid=grid(802816), stream=stream0)
        buf444 = reinterpret_tensor(buf429, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [attn_208], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf442, arg275_1, buf444, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf445 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [attn_208], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf443, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf444, (512, 32, 49), (1568, 49, 1), 0), out=buf445)
        buf448 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [attn_209, attn_210], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_33.run(buf445, arg277_1, arg276_1, buf448, 25088, 49, grid=grid(25088), stream=stream0)
        del arg276_1
        del arg277_1
        buf449 = reinterpret_tensor(buf444, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf444  # reuse
        # Topologically Sorted Source Nodes: [x_723], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf442, arg275_1, buf449, 802816, grid=grid(802816), stream=stream0)
        del arg275_1
        buf450 = reinterpret_tensor(buf443, (512, 49, 32), (1568, 32, 1), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [x_723], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf448, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf449, (512, 49, 32), (1568, 32, 1), 0), out=buf450)
        buf451 = reinterpret_tensor(buf449, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [x_724], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf450, buf451, 802816, grid=grid(802816), stream=stream0)
        buf452 = reinterpret_tensor(buf450, (1568, 512), (512, 1), 0); del buf450  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf451, (1568, 512), (512, 1), 0), reinterpret_tensor(arg278_1, (512, 512), (1, 512), 0), out=buf452)
        del arg278_1
        buf456 = reinterpret_tensor(buf451, (8, 196, 512), (100352, 512, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_93], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf437, buf452, arg279_1, arg280_1, arg281_1, buf456, 1568, 512, grid=grid(1568), stream=stream0)
        del arg280_1
        del arg281_1
        buf457 = reinterpret_tensor(buf435, (1568, 2048), (2048, 1), 0); del buf435  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf456, (1568, 512), (512, 1), 0), reinterpret_tensor(arg282_1, (512, 2048), (1, 512), 0), out=buf457)
        del arg282_1
        buf458 = reinterpret_tensor(buf457, (8, 196, 2048), (401408, 2048, 1), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [x_732], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf458, arg283_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg283_1
        buf459 = reinterpret_tensor(buf456, (1568, 512), (512, 1), 0); del buf456  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf458, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg284_1, (2048, 512), (1, 2048), 0), out=buf459)
        del arg284_1
        buf460 = reinterpret_tensor(buf459, (8, 196, 512), (100352, 512, 1), 0); del buf459  # reuse
        buf461 = buf415; del buf415  # reuse
        buf462 = buf414; del buf414  # reuse
        # Topologically Sorted Source Nodes: [x_736, layer_norm_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf460, buf437, buf452, arg279_1, arg285_1, buf461, buf462, 1568, 512, grid=grid(1568), stream=stream0)
        del arg279_1
        del arg285_1
        buf464 = reinterpret_tensor(buf452, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf452  # reuse
        # Topologically Sorted Source Nodes: [contiguous_172], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf460, buf461, buf462, arg286_1, arg287_1, buf464, 802816, grid=grid(802816), stream=stream0)
        del arg286_1
        del arg287_1
        buf465 = buf442; del buf442  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf464, (1568, 512), (512, 1), 0), reinterpret_tensor(arg289_1, (512, 1536), (1, 512), 0), out=buf465)
        del arg289_1
        buf466 = reinterpret_tensor(buf464, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf464  # reuse
        # Topologically Sorted Source Nodes: [q_87, attn_212], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf465, arg290_1, buf466, 802816, grid=grid(802816), stream=stream0)
        buf467 = reinterpret_tensor(buf437, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [attn_212], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf465, arg290_1, buf467, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf468 = buf445; del buf445  # reuse
        # Topologically Sorted Source Nodes: [attn_212], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf466, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf467, (512, 32, 49), (1568, 49, 1), 0), out=buf468)
        buf472 = buf448; del buf448  # reuse
        # Topologically Sorted Source Nodes: [attn_216], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf468, arg292_1, arg291_1, arg288_1, buf472, 25088, 49, grid=grid(25088), stream=stream0)
        del arg288_1
        del arg291_1
        del arg292_1
        buf473 = reinterpret_tensor(buf467, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [x_739], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf465, arg290_1, buf473, 802816, grid=grid(802816), stream=stream0)
        del arg290_1
        buf474 = reinterpret_tensor(buf466, (512, 49, 32), (1568, 32, 1), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [x_739], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf472, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf473, (512, 49, 32), (1568, 32, 1), 0), out=buf474)
        buf475 = reinterpret_tensor(buf473, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf473  # reuse
        # Topologically Sorted Source Nodes: [x_740], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf474, buf475, 802816, grid=grid(802816), stream=stream0)
        buf476 = reinterpret_tensor(buf474, (1568, 512), (512, 1), 0); del buf474  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf475, (1568, 512), (512, 1), 0), reinterpret_tensor(arg293_1, (512, 512), (1, 512), 0), out=buf476)
        del arg293_1
        buf480 = reinterpret_tensor(buf475, (8, 196, 512), (100352, 512, 1), 0); del buf475  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_95], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf460, buf476, arg294_1, arg295_1, arg296_1, buf480, 1568, 512, grid=grid(1568), stream=stream0)
        del arg295_1
        del arg296_1
        buf481 = reinterpret_tensor(buf458, (1568, 2048), (2048, 1), 0); del buf458  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf480, (1568, 512), (512, 1), 0), reinterpret_tensor(arg297_1, (512, 2048), (1, 512), 0), out=buf481)
        del arg297_1
        buf482 = reinterpret_tensor(buf481, (8, 196, 2048), (401408, 2048, 1), 0); del buf481  # reuse
        # Topologically Sorted Source Nodes: [x_749], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf482, arg298_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg298_1
        buf483 = reinterpret_tensor(buf480, (1568, 512), (512, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf482, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg299_1, (2048, 512), (1, 2048), 0), out=buf483)
        del arg299_1
        buf484 = reinterpret_tensor(buf483, (8, 196, 512), (100352, 512, 1), 0); del buf483  # reuse
        buf488 = reinterpret_tensor(buf413, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [x_753, shifted_x_132, contiguous_176], Original ATen: [aten.add, aten.native_layer_norm, aten.clone]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf484, buf460, buf476, arg294_1, arg300_1, arg301_1, arg302_1, buf488, 1568, 512, grid=grid(1568), stream=stream0)
        del arg294_1
        del arg300_1
        del arg301_1
        del arg302_1
        del buf460
        buf489 = buf465; del buf465  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf488, (1568, 512), (512, 1), 0), reinterpret_tensor(arg303_1, (512, 1536), (1, 512), 0), out=buf489)
        del arg303_1
        buf490 = reinterpret_tensor(buf488, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf488  # reuse
        # Topologically Sorted Source Nodes: [q_89, attn_218], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf489, arg304_1, buf490, 802816, grid=grid(802816), stream=stream0)
        buf491 = reinterpret_tensor(buf476, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [attn_218], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf489, arg304_1, buf491, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf492 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [attn_218], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf490, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf491, (512, 32, 49), (1568, 49, 1), 0), out=buf492)
        buf495 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [attn_219, attn_220], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_33.run(buf492, arg306_1, arg305_1, buf495, 25088, 49, grid=grid(25088), stream=stream0)
        del arg305_1
        del arg306_1
        buf496 = reinterpret_tensor(buf491, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf491  # reuse
        # Topologically Sorted Source Nodes: [x_756], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf489, arg304_1, buf496, 802816, grid=grid(802816), stream=stream0)
        del arg304_1
        buf497 = reinterpret_tensor(buf490, (512, 49, 32), (1568, 32, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [x_756], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf495, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf496, (512, 49, 32), (1568, 32, 1), 0), out=buf497)
        buf498 = reinterpret_tensor(buf496, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf496  # reuse
        # Topologically Sorted Source Nodes: [x_757], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf497, buf498, 802816, grid=grid(802816), stream=stream0)
        buf499 = reinterpret_tensor(buf497, (1568, 512), (512, 1), 0); del buf497  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf498, (1568, 512), (512, 1), 0), reinterpret_tensor(arg307_1, (512, 512), (1, 512), 0), out=buf499)
        del arg307_1
        buf503 = reinterpret_tensor(buf498, (8, 196, 512), (100352, 512, 1), 0); del buf498  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_97], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf484, buf499, arg308_1, arg309_1, arg310_1, buf503, 1568, 512, grid=grid(1568), stream=stream0)
        del arg309_1
        del arg310_1
        buf504 = reinterpret_tensor(buf482, (1568, 2048), (2048, 1), 0); del buf482  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf503, (1568, 512), (512, 1), 0), reinterpret_tensor(arg311_1, (512, 2048), (1, 512), 0), out=buf504)
        del arg311_1
        buf505 = reinterpret_tensor(buf504, (8, 196, 2048), (401408, 2048, 1), 0); del buf504  # reuse
        # Topologically Sorted Source Nodes: [x_765], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf505, arg312_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg312_1
        buf506 = reinterpret_tensor(buf503, (1568, 512), (512, 1), 0); del buf503  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf505, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg313_1, (2048, 512), (1, 2048), 0), out=buf506)
        del arg313_1
        buf507 = reinterpret_tensor(buf506, (8, 196, 512), (100352, 512, 1), 0); del buf506  # reuse
        buf508 = buf462; del buf462  # reuse
        buf509 = buf461; del buf461  # reuse
        # Topologically Sorted Source Nodes: [x_769, layer_norm_98], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf507, buf484, buf499, arg308_1, arg314_1, buf508, buf509, 1568, 512, grid=grid(1568), stream=stream0)
        del arg308_1
        del arg314_1
        buf511 = reinterpret_tensor(buf499, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf499  # reuse
        # Topologically Sorted Source Nodes: [contiguous_180], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf507, buf508, buf509, arg315_1, arg316_1, buf511, 802816, grid=grid(802816), stream=stream0)
        del arg315_1
        del arg316_1
        del buf508
        del buf509
        buf512 = buf489; del buf489  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf511, (1568, 512), (512, 1), 0), reinterpret_tensor(arg318_1, (512, 1536), (1, 512), 0), out=buf512)
        del arg318_1
        buf513 = reinterpret_tensor(buf511, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [q_91, attn_222], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_31.run(buf512, arg319_1, buf513, 802816, grid=grid(802816), stream=stream0)
        buf514 = reinterpret_tensor(buf484, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf484  # reuse
        # Topologically Sorted Source Nodes: [attn_222], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf512, arg319_1, buf514, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf515 = buf492; del buf492  # reuse
        # Topologically Sorted Source Nodes: [attn_222], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf513, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf514, (512, 32, 49), (1568, 49, 1), 0), out=buf515)
        buf519 = buf495; del buf495  # reuse
        # Topologically Sorted Source Nodes: [attn_226], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf515, arg321_1, arg320_1, arg317_1, buf519, 25088, 49, grid=grid(25088), stream=stream0)
        del arg317_1
        del arg320_1
        del arg321_1
        del buf515
        buf520 = reinterpret_tensor(buf514, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [x_772], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf512, arg319_1, buf520, 802816, grid=grid(802816), stream=stream0)
        del arg319_1
        del buf512
        buf521 = reinterpret_tensor(buf513, (512, 49, 32), (1568, 32, 1), 0); del buf513  # reuse
        # Topologically Sorted Source Nodes: [x_772], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf519, (512, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf520, (512, 49, 32), (1568, 32, 1), 0), out=buf521)
        del buf519
        buf522 = reinterpret_tensor(buf520, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [x_773], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf521, buf522, 802816, grid=grid(802816), stream=stream0)
        buf523 = reinterpret_tensor(buf521, (1568, 512), (512, 1), 0); del buf521  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf522, (1568, 512), (512, 1), 0), reinterpret_tensor(arg322_1, (512, 512), (1, 512), 0), out=buf523)
        del arg322_1
        buf527 = reinterpret_tensor(buf522, (8, 196, 512), (100352, 512, 1), 0); del buf522  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_99], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf507, buf523, arg323_1, arg324_1, arg325_1, buf527, 1568, 512, grid=grid(1568), stream=stream0)
        del arg324_1
        del arg325_1
        buf528 = reinterpret_tensor(buf505, (1568, 2048), (2048, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf527, (1568, 512), (512, 1), 0), reinterpret_tensor(arg326_1, (512, 2048), (1, 512), 0), out=buf528)
        del arg326_1
        buf529 = reinterpret_tensor(buf528, (8, 196, 2048), (401408, 2048, 1), 0); del buf528  # reuse
        # Topologically Sorted Source Nodes: [x_782], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf529, arg327_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg327_1
        buf530 = reinterpret_tensor(buf527, (1568, 512), (512, 1), 0); del buf527  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf529, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg328_1, (2048, 512), (1, 2048), 0), out=buf530)
        del arg328_1
        del buf529
        buf531 = reinterpret_tensor(buf530, (8, 7, 7, 2, 2, 512), (100352, 14336, 1024, 512, 7168, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [x_789], Original ATen: [aten.clone]
        triton_poi_fused_clone_43.run(buf531, buf507, buf523, arg323_1, arg329_1, 802816, grid=grid(802816), stream=stream0)
        del arg323_1
        del arg329_1
        del buf507
        buf535 = reinterpret_tensor(buf523, (8, 7, 7, 2048), (100352, 14336, 2048, 1), 0); del buf523  # reuse
        # Topologically Sorted Source Nodes: [x_790], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_44.run(buf531, arg330_1, arg331_1, buf535, 392, 2048, grid=grid(392), stream=stream0)
        del arg330_1
        del arg331_1
        del buf531
        buf536 = empty_strided_cuda((392, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_791], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (392, 2048), (2048, 1), 0), reinterpret_tensor(arg332_1, (2048, 1024), (1, 2048), 0), out=buf536)
        del arg332_1
        del buf535
        buf540 = empty_strided_cuda((8, 7, 7, 1024), (50176, 7168, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [shifted_x_138], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_45.run(buf536, arg333_1, arg334_1, buf540, 392, 1024, grid=grid(392), stream=stream0)
        del arg333_1
        del arg334_1
        buf541 = empty_strided_cuda((392, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf540, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg335_1, (1024, 3072), (1, 1024), 0), out=buf541)
        del arg335_1
        buf542 = reinterpret_tensor(buf540, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf540  # reuse
        # Topologically Sorted Source Nodes: [q_93, attn_228], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_46.run(buf541, arg336_1, buf542, 401408, grid=grid(401408), stream=stream0)
        buf543 = empty_strided_cuda((8, 32, 32, 49), (50176, 1568, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_228], Original ATen: [aten.clone]
        triton_poi_fused_clone_47.run(buf541, arg336_1, buf543, 8192, 49, grid=grid(8192, 49), stream=stream0)
        buf544 = empty_strided_cuda((256, 49, 49), (2401, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_228], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf542, (256, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf543, (256, 32, 49), (1568, 49, 1), 0), out=buf544)
        buf547 = empty_strided_cuda((8, 32, 49, 49), (77824, 2432, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_229, attn_230], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_48.run(buf544, arg338_1, arg337_1, buf547, 12544, 49, grid=grid(12544), stream=stream0)
        del arg337_1
        del arg338_1
        buf548 = reinterpret_tensor(buf543, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [x_793], Original ATen: [aten.clone]
        triton_poi_fused_clone_49.run(buf541, arg336_1, buf548, 401408, grid=grid(401408), stream=stream0)
        del arg336_1
        buf549 = reinterpret_tensor(buf542, (256, 49, 32), (1568, 32, 1), 0); del buf542  # reuse
        # Topologically Sorted Source Nodes: [x_793], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf547, (256, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf548, (256, 49, 32), (1568, 32, 1), 0), out=buf549)
        buf550 = reinterpret_tensor(buf548, (8, 49, 32, 32), (50176, 1024, 32, 1), 0); del buf548  # reuse
        # Topologically Sorted Source Nodes: [x_794], Original ATen: [aten.clone]
        triton_poi_fused_clone_50.run(buf549, buf550, 401408, grid=grid(401408), stream=stream0)
        buf551 = reinterpret_tensor(buf549, (392, 1024), (1024, 1), 0); del buf549  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf550, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg339_1, (1024, 1024), (1, 1024), 0), out=buf551)
        del arg339_1
        buf555 = reinterpret_tensor(buf550, (8, 49, 1024), (50176, 1024, 1), 0); del buf550  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_102], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_51.run(buf536, buf551, arg340_1, arg341_1, arg342_1, buf555, 392, 1024, grid=grid(392), stream=stream0)
        del arg341_1
        del arg342_1
        buf556 = reinterpret_tensor(buf107, (392, 4096), (4096, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf555, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg343_1, (1024, 4096), (1, 1024), 0), out=buf556)
        del arg343_1
        buf557 = reinterpret_tensor(buf556, (8, 49, 4096), (200704, 4096, 1), 0); del buf556  # reuse
        # Topologically Sorted Source Nodes: [x_802], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_52.run(buf557, arg344_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg344_1
        buf558 = reinterpret_tensor(buf555, (392, 1024), (1024, 1), 0); del buf555  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf557, (392, 4096), (4096, 1), 0), reinterpret_tensor(arg345_1, (4096, 1024), (1, 4096), 0), out=buf558)
        del arg345_1
        buf559 = reinterpret_tensor(buf558, (8, 49, 1024), (50176, 1024, 1), 0); del buf558  # reuse
        buf563 = empty_strided_cuda((8, 7, 7, 1024), (50176, 7168, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_806, shifted_x_141], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_53.run(buf559, buf536, buf551, arg340_1, arg346_1, arg347_1, arg348_1, buf563, 392, 1024, grid=grid(392), stream=stream0)
        del arg340_1
        del arg346_1
        del arg347_1
        del arg348_1
        del buf536
        buf564 = buf541; del buf541  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf563, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg349_1, (1024, 3072), (1, 1024), 0), out=buf564)
        del arg349_1
        buf565 = reinterpret_tensor(buf563, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf563  # reuse
        # Topologically Sorted Source Nodes: [q_95, attn_232], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_46.run(buf564, arg350_1, buf565, 401408, grid=grid(401408), stream=stream0)
        buf566 = reinterpret_tensor(buf551, (8, 32, 32, 49), (50176, 1568, 49, 1), 0); del buf551  # reuse
        # Topologically Sorted Source Nodes: [attn_232], Original ATen: [aten.clone]
        triton_poi_fused_clone_47.run(buf564, arg350_1, buf566, 8192, 49, grid=grid(8192, 49), stream=stream0)
        buf567 = buf544; del buf544  # reuse
        # Topologically Sorted Source Nodes: [attn_232], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf565, (256, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf566, (256, 32, 49), (1568, 49, 1), 0), out=buf567)
        buf570 = buf547; del buf547  # reuse
        # Topologically Sorted Source Nodes: [attn_233, attn_234], Original ATen: [aten.add, aten._softmax]
        triton_per_fused__softmax_add_48.run(buf567, arg352_1, arg351_1, buf570, 12544, 49, grid=grid(12544), stream=stream0)
        del arg351_1
        del arg352_1
        del buf567
        buf571 = reinterpret_tensor(buf566, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf566  # reuse
        # Topologically Sorted Source Nodes: [x_809], Original ATen: [aten.clone]
        triton_poi_fused_clone_49.run(buf564, arg350_1, buf571, 401408, grid=grid(401408), stream=stream0)
        del arg350_1
        del buf564
        buf572 = reinterpret_tensor(buf565, (256, 49, 32), (1568, 32, 1), 0); del buf565  # reuse
        # Topologically Sorted Source Nodes: [x_809], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf570, (256, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf571, (256, 49, 32), (1568, 32, 1), 0), out=buf572)
        del buf570
        buf573 = reinterpret_tensor(buf571, (8, 49, 32, 32), (50176, 1024, 32, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [x_810], Original ATen: [aten.clone]
        triton_poi_fused_clone_50.run(buf572, buf573, 401408, grid=grid(401408), stream=stream0)
        buf574 = reinterpret_tensor(buf572, (392, 1024), (1024, 1), 0); del buf572  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf573, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg353_1, (1024, 1024), (1, 1024), 0), out=buf574)
        del arg353_1
        buf578 = reinterpret_tensor(buf573, (8, 49, 1024), (50176, 1024, 1), 0); del buf573  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_104], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_51.run(buf559, buf574, arg354_1, arg355_1, arg356_1, buf578, 392, 1024, grid=grid(392), stream=stream0)
        del arg355_1
        del arg356_1
        buf579 = reinterpret_tensor(buf557, (392, 4096), (4096, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf578, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg357_1, (1024, 4096), (1, 1024), 0), out=buf579)
        del arg357_1
        buf580 = reinterpret_tensor(buf579, (8, 49, 4096), (200704, 4096, 1), 0); del buf579  # reuse
        # Topologically Sorted Source Nodes: [x_818], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_52.run(buf580, arg358_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg358_1
        buf581 = reinterpret_tensor(buf578, (392, 1024), (1024, 1), 0); del buf578  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf580, (392, 4096), (4096, 1), 0), reinterpret_tensor(arg359_1, (4096, 1024), (1, 4096), 0), out=buf581)
        del arg359_1
        del buf580
        buf582 = reinterpret_tensor(buf581, (8, 49, 1024), (50176, 1024, 1), 0); del buf581  # reuse
        buf583 = empty_strided_cuda((8, 7, 7, 1), (49, 7, 1, 392), torch.float32)
        buf584 = empty_strided_cuda((8, 7, 7, 1), (49, 7, 1, 392), torch.float32)
        # Topologically Sorted Source Nodes: [x_822, x_824], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_54.run(buf582, buf559, buf574, arg354_1, arg360_1, buf583, buf584, 392, 1024, grid=grid(392), stream=stream0)
        del arg354_1
        del arg360_1
        del buf559
        del buf574
        buf587 = empty_strided_cuda((8, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_824, x_825], Original ATen: [aten.native_layer_norm, aten.mean]
        triton_per_fused_mean_native_layer_norm_55.run(buf582, buf583, buf584, arg361_1, arg362_1, buf587, 8192, 49, grid=grid(8192), stream=stream0)
        del arg361_1
        del arg362_1
        del buf582
        del buf583
        del buf584
        buf588 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_824, x_825, x_827], Original ATen: [aten.native_layer_norm, aten.mean, aten.addmm]
        extern_kernels.addmm(arg364_1, buf587, reinterpret_tensor(arg363_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf588)
        del arg363_1
        del arg364_1
        del buf587
    return (buf588, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((169, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg11_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((169, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg26_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((169, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg43_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((16, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((169, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg58_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg75_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
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
    arg102_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg104_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg119_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg133_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg148_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
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
    arg172_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg177_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg191_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg206_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg220_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg235_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg249_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg264_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg278_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg293_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg307_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg322_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((169, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg339_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((169, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg353_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('swin_base_patch4_window7_224', benchmark_compiled_module)
