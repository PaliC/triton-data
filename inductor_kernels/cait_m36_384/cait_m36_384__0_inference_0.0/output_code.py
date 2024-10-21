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


# kernel path: /tmp/torchinductor_sahanp/7p/c7p7mscen2qdo2jikv55t5olhchkxau6tvycnmbcrogvlfw4oged.py
# Topologically Sorted Source Nodes: [x_381, layer_norm_77], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_77 => clone_513, var_mean_77
#   x_381 => add_341
# Graph fragment:
#   %add_341 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_490, %arg3_1), kwargs = {})
#   %clone_513 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_341,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_77 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_513, [2]), kwargs = {correction: 0, keepdim: True})
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
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 27648
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 6
    x1 = (xindex // 6) % 576
    x2 = (xindex // 3456)
    x5 = xindex % 3456
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (576*r3) + (73728*x0) + (442368*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x6), tmp6, xmask)
    tl.store(out_ptr1 + (x6), tmp7, xmask)
    tl.store(out_ptr2 + (x6), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jj/cjjip6ny3anvqgjtkibe62js5tz2xwvjbxkgp3a4574u4o6gv3ys.py
# Topologically Sorted Source Nodes: [x_381, layer_norm_77], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_77 => clone_513, var_mean_77
#   x_381 => add_341
# Graph fragment:
#   %add_341 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_490, %arg3_1), kwargs = {})
#   %clone_513 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_341,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_77 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_513, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_1 = async_compile.triton('triton_per_fused_add_native_layer_norm_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (6*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/oe/coe2vnoeu42as7zl56cf5nsfgd36gbqmklkews6bkeo4hc2lig3t.py
# Topologically Sorted Source Nodes: [x_381, layer_norm_77], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_77 => add_342, add_343, clone_513, mul_380, mul_381, rsqrt_77, sub_113, var_mean_77
#   x_381 => add_341
# Graph fragment:
#   %add_341 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_490, %arg3_1), kwargs = {})
#   %clone_513 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_341,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_77 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_513, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_513, %getitem_163), kwargs = {})
#   %add_342 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_162, 1e-06), kwargs = {})
#   %rsqrt_77 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_342,), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %rsqrt_77), kwargs = {})
#   %mul_381 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_380, %arg5_1), kwargs = {})
#   %add_343 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_381, %arg6_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_2 = async_compile.triton('triton_poi_fused_add_native_layer_norm_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 576
    y1 = (yindex // 576)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (576*x2) + (442368*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (768*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 768.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5d/c5dogld5trwmuckyu2bp7cupsculoxfkujmzzjgtnj3hpyngdfgj.py
# Topologically Sorted Source Nodes: [q_38, attn_180], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   attn_180 => clone_514
#   q_38 => mul_382
# Graph fragment:
#   %mul_382 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_111, 0.14433756729740643), kwargs = {})
#   %clone_514 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_145,), kwargs = {memory_format: torch.contiguous_format})
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
    xnumel = 3538944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 48
    x1 = (xindex // 48) % 576
    x2 = (xindex // 27648) % 16
    x3 = (xindex // 442368)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (2304*x1) + (1327104*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (48*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.14433756729740643
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ke/ckeovgw2pbczfmqj5fpbrvv7xllqw6dox65rr4xzlbk3saipjtpd.py
# Topologically Sorted Source Nodes: [attn_180], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_180 => clone_515
# Graph fragment:
#   %clone_515 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_146,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (768 + y0 + (2304*x2) + (1327104*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (768 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (576*y3)), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zo/czoq57dj5fmncdq5bectr2npet6kklnzyabg3lplz3kbnrtuszs6.py
# Topologically Sorted Source Nodes: [linear_230], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   linear_230 => clone_516
# Graph fragment:
#   %clone_516 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_494,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4194304, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2654208
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 331776
    y1 = (yindex // 331776)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (331776*x2) + (5308416*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h2/ch2odrzfv6z6jiubfioq46uyoyp4xvd5qim2eypxxyujaba6x7iw.py
# Topologically Sorted Source Nodes: [attn_182], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_182 => amax_36, clone_517, exp_36, sub_114, sum_37
# Graph fragment:
#   %clone_517 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_496,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_36 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_517, [-1], True), kwargs = {})
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_517, %amax_36), kwargs = {})
#   %exp_36 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_114,), kwargs = {})
#   %sum_37 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_36, [-1], True), kwargs = {})
triton_red_fused__softmax_6 = async_compile.triton('triton_red_fused__softmax_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[131072, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 73728
    rnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (9216*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp6 = tl.load(in_ptr0 + (x0 + (16*r2) + (9216*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp6 + tmp1
        tmp8 = tmp7 - tmp4
        tmp9 = tl_math.exp(tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3j/c3joundggibooaozoqhtyt7tww27shs4itonrkctl3idctuygysj.py
# Topologically Sorted Source Nodes: [linear_231], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   linear_231 => clone_518
# Graph fragment:
#   %clone_518 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_497,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 42467328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 16
    x2 = (xindex // 9216)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp7 = tmp5 / tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/s7/cs7yx75nskpokgobye2q5ougvuohsxgl7sy4nyvxtvuznxikt2g2.py
# Topologically Sorted Source Nodes: [matmul_73], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_73 => clone_520
# Graph fragment:
#   %clone_520 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_147,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_8 = async_compile.triton('triton_poi_fused_clone_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[128, 524288], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 331776
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x2) + (5308416*y1)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (331776*y3)), tmp2, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/x3/cx3w57b6vadhkbevvxprzu4xyfkpgkskrimykupfjxh4gji2fctv.py
# Topologically Sorted Source Nodes: [matmul_73], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_73 => clone_521
# Graph fragment:
#   %clone_521 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_148,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_poi_fused_clone_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3538944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 48
    x1 = (xindex // 48) % 576
    x2 = (xindex // 27648) % 16
    x3 = (xindex // 442368)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + (48*x2) + (2304*x1) + (1327104*x3)), None)
    tmp1 = tl.load(in_ptr1 + (1536 + x0 + (48*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fh/cfh5wcho4ncilowjx76i2rk7urt6xpv2iuwoebbs2fz32dalncn5.py
# Topologically Sorted Source Nodes: [x_383], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_383 => clone_522
# Graph fragment:
#   %clone_522 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_500,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_10 = async_compile.triton('triton_poi_fused_clone_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3538944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 48
    x1 = (xindex // 48) % 16
    x2 = (xindex // 768) % 576
    x3 = (xindex // 442368)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (27648*x1) + (442368*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ly/cly4gw6si5gchx7rciubvynxriowhi6ae5n3wsxvey4324jv5ee6.py
# Topologically Sorted Source Nodes: [x_381, mul_113, x_386, layer_norm_78], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_78 => add_347, add_348, clone_524, mul_384, mul_385, rsqrt_78, sub_115, var_mean_78
#   mul_113 => mul_383
#   x_381 => add_341
#   x_386 => add_346
# Graph fragment:
#   %add_341 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_490, %arg3_1), kwargs = {})
#   %mul_383 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, %view_765), kwargs = {})
#   %add_346 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_341, %mul_383), kwargs = {})
#   %clone_524 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_346,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_78 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_524, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_524, %getitem_165), kwargs = {})
#   %add_347 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_164, 1e-06), kwargs = {})
#   %rsqrt_78 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_347,), kwargs = {})
#   %mul_384 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %rsqrt_78), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_384, %arg16_1), kwargs = {})
#   %add_348 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_385, %arg17_1), kwargs = {})
triton_red_fused_add_mul_native_layer_norm_11 = async_compile.triton('triton_red_fused_add_mul_native_layer_norm_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_native_layer_norm_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4608
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 576
    x1 = (xindex // 576)
    x3 = xindex
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (576*r2) + (442368*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tmp4 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
        tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp10, rmask & xmask)
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
        tmp15 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp15 - tmp12
        tmp17 = 768.0
        tmp18 = tmp13 / tmp17
        tmp19 = 1e-06
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp24 = tmp22 * tmp23
        tmp26 = tmp24 + tmp25
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp26, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ic/cicv56htg6pfo7xir7cihd5vbkzuu2543e2m47frxse64tnmsm2r.py
# Topologically Sorted Source Nodes: [x_388], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_388 => add_349, erf_38, mul_386, mul_387, mul_388
# Graph fragment:
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_767, 0.5), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_767, 0.7071067811865476), kwargs = {})
#   %erf_38 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_387,), kwargs = {})
#   %add_349 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_38, 1), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_386, %add_349), kwargs = {})
triton_poi_fused_gelu_12 = async_compile.triton('triton_poi_fused_gelu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14155776
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


# kernel path: /tmp/torchinductor_sahanp/ug/cugby3skfdehn4fr3wg5oszyldxluwyb7gxjedl6if2gmkutxqnd.py
# Topologically Sorted Source Nodes: [mul_114, x_392, layer_norm_79], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_79 => add_351, add_352, clone_527, mul_390, mul_391, rsqrt_79, sub_116, var_mean_79
#   mul_114 => mul_389
#   x_392 => add_350
# Graph fragment:
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg15_1, %view_769), kwargs = {})
#   %add_350 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_346, %mul_389), kwargs = {})
#   %clone_527 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_350,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_79 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_527, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_116 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_527, %getitem_167), kwargs = {})
#   %add_351 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_166, 1e-06), kwargs = {})
#   %rsqrt_79 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_351,), kwargs = {})
#   %mul_390 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_116, %rsqrt_79), kwargs = {})
#   %mul_391 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_390, %arg23_1), kwargs = {})
#   %add_352 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_391, %arg24_1), kwargs = {})
triton_per_fused_add_mul_native_layer_norm_13 = async_compile.triton('triton_per_fused_add_mul_native_layer_norm_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mul_native_layer_norm_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 4608
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp33, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/we/cwejoivd2fkogw4yywb2j6p2cbledu4xmjddlcibfuwqtttzwvpf.py
# Topologically Sorted Source Nodes: [mul_114, x_392, mul_116, x_396, layer_norm_80], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_80 => add_356, add_357, clone_538, mul_394, mul_395, rsqrt_80, sub_118, var_mean_80
#   mul_114 => mul_389
#   mul_116 => mul_393
#   x_392 => add_350
#   x_396 => add_355
# Graph fragment:
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg15_1, %view_769), kwargs = {})
#   %add_350 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_346, %mul_389), kwargs = {})
#   %mul_393 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg22_1, %view_785), kwargs = {})
#   %add_355 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_350, %mul_393), kwargs = {})
#   %clone_538 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_355,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_80 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_538, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_118 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_538, %getitem_169), kwargs = {})
#   %add_356 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_168, 1e-06), kwargs = {})
#   %rsqrt_80 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_356,), kwargs = {})
#   %mul_394 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_118, %rsqrt_80), kwargs = {})
#   %mul_395 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_394, %arg34_1), kwargs = {})
#   %add_357 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_395, %arg35_1), kwargs = {})
triton_per_fused_add_mul_native_layer_norm_14 = async_compile.triton('triton_per_fused_add_mul_native_layer_norm_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 9, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mul_native_layer_norm_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, rnumel):
    xnumel = 4608
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp12, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp39, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/24/c24hpt2x7cyz6akz2z67ktfwezyljtmckx5syleib7riipteqeoq.py
# Topologically Sorted Source Nodes: [u_2, layer_norm_149], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_149 => add_666, add_667, mul_740, mul_741, rsqrt_149, sub_221, var_mean_149
#   u_2 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_289, %add_665], 1), kwargs = {})
#   %var_mean_149 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_221 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_3, %getitem_307), kwargs = {})
#   %add_666 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_306, 1e-06), kwargs = {})
#   %rsqrt_149 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_666,), kwargs = {})
#   %mul_740 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_221, %rsqrt_149), kwargs = {})
#   %mul_741 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_740, %arg654_1), kwargs = {})
#   %add_667 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_741, %arg655_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_15 = async_compile.triton('triton_per_fused_cat_native_layer_norm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 4616
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 577
    r2 = rindex
    x1 = (xindex // 577)
    x3 = xindex
    tmp42 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 577, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (768*((-1) + x0)) + (442368*x1)), rmask & tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr3 + (r2 + (768*((-1) + x0)) + (442368*x1)), rmask & tmp6, other=0.0)
    tmp12 = tl.load(in_ptr4 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 * tmp13
    tmp15 = tmp9 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp6, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp5, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 768, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tmp18 - tmp28
    tmp36 = 768.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp18, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp45, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mb/cmbavhpr7kdhxqpcmo7q4w3jvkpy5fmlzno5iqrt33qufbw7nkuk.py
# Topologically Sorted Source Nodes: [mul_220, x_cls_16, layer_norm_150], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_150 => add_669, add_670, mul_743, mul_744, rsqrt_150, sub_222, var_mean_150
#   mul_220 => mul_742
#   x_cls_16 => add_668
# Graph fragment:
#   %mul_742 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg653_1, %view_1479), kwargs = {})
#   %add_668 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_289, %mul_742), kwargs = {})
#   %var_mean_150 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_668, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_222 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_668, %getitem_313), kwargs = {})
#   %add_669 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_312, 1e-06), kwargs = {})
#   %rsqrt_150 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_669,), kwargs = {})
#   %mul_743 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_222, %rsqrt_150), kwargs = {})
#   %mul_744 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_743, %arg665_1), kwargs = {})
#   %add_670 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_744, %arg666_1), kwargs = {})
triton_per_fused_add_mul_native_layer_norm_16 = async_compile.triton('triton_per_fused_add_mul_native_layer_norm_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mul_native_layer_norm_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 8
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
    tmp0 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp33, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/my/cmyfe2tj3q2wojfeexfmusjnwfiig77v4uahyeo7rrcjbefg2wr5.py
# Topologically Sorted Source Nodes: [x_744], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_744 => add_671, erf_74, mul_745, mul_746, mul_747
# Graph fragment:
#   %mul_745 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1481, 0.5), kwargs = {})
#   %mul_746 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1481, 0.7071067811865476), kwargs = {})
#   %erf_74 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_746,), kwargs = {})
#   %add_671 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_74, 1), kwargs = {})
#   %mul_747 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_745, %add_671), kwargs = {})
triton_poi_fused_gelu_17 = async_compile.triton('triton_poi_fused_gelu_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
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


# kernel path: /tmp/torchinductor_sahanp/ij/cijxvcn2rsix6ryk7d7rvp7mikoynqgpwpfctxdedvkogndle6jz.py
# Topologically Sorted Source Nodes: [mul_220, x_cls_16, mul_221, x_cls_17], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_220 => mul_742
#   mul_221 => mul_748
#   x_cls_16 => add_668
#   x_cls_17 => add_672
# Graph fragment:
#   %mul_742 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg653_1, %view_1479), kwargs = {})
#   %add_668 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_289, %mul_742), kwargs = {})
#   %mul_748 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg664_1, %view_1483), kwargs = {})
#   %add_672 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_668, %mul_748), kwargs = {})
triton_poi_fused_add_mul_18 = async_compile.triton('triton_poi_fused_add_mul_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask)
    tmp3 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask)
    tmp9 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp6 + tmp11
    tl.store(out_ptr0 + (x0 + (443136*x1)), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ys/cysjy2xbqjw26wtyrcdlochebzsr4owpxrkpilwbaulas5rhzwnp.py
# Topologically Sorted Source Nodes: [mul_219, x_742], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_219 => mul_739
#   x_742 => add_665
# Graph fragment:
#   %mul_739 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg645_1, %view_1469), kwargs = {})
#   %add_665 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_661, %mul_739), kwargs = {})
triton_poi_fused_add_mul_19 = async_compile.triton('triton_poi_fused_add_mul_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3538944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 768
    x2 = (xindex // 442368)
    x4 = xindex % 442368
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x3), None)
    tmp3 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(out_ptr0 + (x4 + (443136*x2)), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rl/crlddug2xzua5xyjzy2vjlq6xksq3qpdyli72hqrl3bx6hsiyxje.py
# Topologically Sorted Source Nodes: [layer_norm_151], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_151 => add_673, add_674, mul_749, mul_750, rsqrt_151, sub_223, var_mean_151
# Graph fragment:
#   %var_mean_151 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_223 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_4, %getitem_315), kwargs = {})
#   %add_673 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_314, 1e-06), kwargs = {})
#   %rsqrt_151 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_673,), kwargs = {})
#   %mul_749 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_223, %rsqrt_151), kwargs = {})
#   %mul_750 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_749, %arg672_1), kwargs = {})
#   %add_674 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_750, %arg673_1), kwargs = {})
triton_per_fused_native_layer_norm_20 = async_compile.triton('triton_per_fused_native_layer_norm_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_20(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 4616
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
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 768.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qo/cqo5v5fknzxivm3e5eai7k7jxi33qmw74v42zpicjgrbat5eyvty.py
# Topologically Sorted Source Nodes: [mul_222, x_cls_22, layer_norm_152], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_152 => add_676, add_677, mul_752, mul_753, rsqrt_152, sub_224, var_mean_152
#   mul_222 => mul_751
#   x_cls_22 => add_675
# Graph fragment:
#   %mul_751 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg671_1, %view_1493), kwargs = {})
#   %add_675 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_672, %mul_751), kwargs = {})
#   %var_mean_152 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_675, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_224 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_675, %getitem_321), kwargs = {})
#   %add_676 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_320, 1e-06), kwargs = {})
#   %rsqrt_152 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_676,), kwargs = {})
#   %mul_752 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_224, %rsqrt_152), kwargs = {})
#   %mul_753 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_752, %arg683_1), kwargs = {})
#   %add_677 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_753, %arg684_1), kwargs = {})
triton_per_fused_add_mul_native_layer_norm_21 = async_compile.triton('triton_per_fused_add_mul_native_layer_norm_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mul_native_layer_norm_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 8
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
    tmp0 = tl.load(in_ptr0 + (r1 + (443136*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp1 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-06
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp33, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jj/cjjsy4acqahk3bio5djzpmqpgmmw5g4ucv4p7ywjcysmxx4kglcg.py
# Topologically Sorted Source Nodes: [x_753, x_754], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_753 => cat_5
#   x_754 => var_mean_153
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_679, %add_665], 1), kwargs = {})
#   %var_mean_153 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_5, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_cat_native_layer_norm_22 = async_compile.triton('triton_per_fused_cat_native_layer_norm_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 4616
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 577
    r2 = rindex
    x1 = (xindex // 577)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (443136*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r2 + (768*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tmp5 + tmp10
    tmp12 = tl.load(in_ptr4 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr5 + (r2 + (768*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr6 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 * tmp15
    tmp17 = tmp11 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 577, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.load(in_ptr7 + (r2 + (768*((-1) + x0)) + (443136*x1)), rmask & tmp20, other=0.0)
    tmp24 = tl.where(tmp4, tmp19, tmp23)
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tl.full([1], 768, tl.int32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 / tmp33
    tmp35 = tmp25 - tmp34
    tmp36 = tmp35 * tmp35
    tmp37 = tl.broadcast_to(tmp36, [RBLOCK])
    tmp39 = tl.where(rmask, tmp37, 0)
    tmp40 = triton_helpers.promote_to_tensor(tl.sum(tmp39, 0))
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp24, rmask)
    tl.store(out_ptr1 + (x3), tmp34, None)
    tl.store(out_ptr2 + (x3), tmp40, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/im/cimqau4iz5vnjegu3jrkwvxoxohn35qxozknumsabtnsbhxoqmku.py
# Topologically Sorted Source Nodes: [x_756], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_756 => clone_1023
# Graph fragment:
#   %clone_1023 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%select_221,), kwargs = {})
triton_poi_fused_clone_23 = async_compile.triton('triton_poi_fused_clone_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (443136*x1)), xmask)
    tmp1 = tl.load(in_ptr1 + (577*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (577*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 384, 384), (442368, 147456, 384, 1))
    assert_size_stride(arg1_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (1, 576, 768), (442368, 768, 1))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (2304, 768), (768, 1))
    assert_size_stride(arg8_1, (2304, ), (1, ))
    assert_size_stride(arg9_1, (16, 16), (16, 1))
    assert_size_stride(arg10_1, (16, ), (1, ))
    assert_size_stride(arg11_1, (16, 16), (16, 1))
    assert_size_stride(arg12_1, (16, ), (1, ))
    assert_size_stride(arg13_1, (768, 768), (768, 1))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (3072, 768), (768, 1))
    assert_size_stride(arg19_1, (3072, ), (1, ))
    assert_size_stride(arg20_1, (768, 3072), (3072, 1))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (2304, 768), (768, 1))
    assert_size_stride(arg26_1, (2304, ), (1, ))
    assert_size_stride(arg27_1, (16, 16), (16, 1))
    assert_size_stride(arg28_1, (16, ), (1, ))
    assert_size_stride(arg29_1, (16, 16), (16, 1))
    assert_size_stride(arg30_1, (16, ), (1, ))
    assert_size_stride(arg31_1, (768, 768), (768, 1))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (3072, 768), (768, 1))
    assert_size_stride(arg37_1, (3072, ), (1, ))
    assert_size_stride(arg38_1, (768, 3072), (3072, 1))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (2304, 768), (768, 1))
    assert_size_stride(arg44_1, (2304, ), (1, ))
    assert_size_stride(arg45_1, (16, 16), (16, 1))
    assert_size_stride(arg46_1, (16, ), (1, ))
    assert_size_stride(arg47_1, (16, 16), (16, 1))
    assert_size_stride(arg48_1, (16, ), (1, ))
    assert_size_stride(arg49_1, (768, 768), (768, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (3072, 768), (768, 1))
    assert_size_stride(arg55_1, (3072, ), (1, ))
    assert_size_stride(arg56_1, (768, 3072), (3072, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (2304, 768), (768, 1))
    assert_size_stride(arg62_1, (2304, ), (1, ))
    assert_size_stride(arg63_1, (16, 16), (16, 1))
    assert_size_stride(arg64_1, (16, ), (1, ))
    assert_size_stride(arg65_1, (16, 16), (16, 1))
    assert_size_stride(arg66_1, (16, ), (1, ))
    assert_size_stride(arg67_1, (768, 768), (768, 1))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (3072, 768), (768, 1))
    assert_size_stride(arg73_1, (3072, ), (1, ))
    assert_size_stride(arg74_1, (768, 3072), (3072, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (2304, 768), (768, 1))
    assert_size_stride(arg80_1, (2304, ), (1, ))
    assert_size_stride(arg81_1, (16, 16), (16, 1))
    assert_size_stride(arg82_1, (16, ), (1, ))
    assert_size_stride(arg83_1, (16, 16), (16, 1))
    assert_size_stride(arg84_1, (16, ), (1, ))
    assert_size_stride(arg85_1, (768, 768), (768, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (3072, 768), (768, 1))
    assert_size_stride(arg91_1, (3072, ), (1, ))
    assert_size_stride(arg92_1, (768, 3072), (3072, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (2304, 768), (768, 1))
    assert_size_stride(arg98_1, (2304, ), (1, ))
    assert_size_stride(arg99_1, (16, 16), (16, 1))
    assert_size_stride(arg100_1, (16, ), (1, ))
    assert_size_stride(arg101_1, (16, 16), (16, 1))
    assert_size_stride(arg102_1, (16, ), (1, ))
    assert_size_stride(arg103_1, (768, 768), (768, 1))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (3072, 768), (768, 1))
    assert_size_stride(arg109_1, (3072, ), (1, ))
    assert_size_stride(arg110_1, (768, 3072), (3072, 1))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (2304, 768), (768, 1))
    assert_size_stride(arg116_1, (2304, ), (1, ))
    assert_size_stride(arg117_1, (16, 16), (16, 1))
    assert_size_stride(arg118_1, (16, ), (1, ))
    assert_size_stride(arg119_1, (16, 16), (16, 1))
    assert_size_stride(arg120_1, (16, ), (1, ))
    assert_size_stride(arg121_1, (768, 768), (768, 1))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (3072, 768), (768, 1))
    assert_size_stride(arg127_1, (3072, ), (1, ))
    assert_size_stride(arg128_1, (768, 3072), (3072, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (2304, 768), (768, 1))
    assert_size_stride(arg134_1, (2304, ), (1, ))
    assert_size_stride(arg135_1, (16, 16), (16, 1))
    assert_size_stride(arg136_1, (16, ), (1, ))
    assert_size_stride(arg137_1, (16, 16), (16, 1))
    assert_size_stride(arg138_1, (16, ), (1, ))
    assert_size_stride(arg139_1, (768, 768), (768, 1))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (3072, 768), (768, 1))
    assert_size_stride(arg145_1, (3072, ), (1, ))
    assert_size_stride(arg146_1, (768, 3072), (3072, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (2304, 768), (768, 1))
    assert_size_stride(arg152_1, (2304, ), (1, ))
    assert_size_stride(arg153_1, (16, 16), (16, 1))
    assert_size_stride(arg154_1, (16, ), (1, ))
    assert_size_stride(arg155_1, (16, 16), (16, 1))
    assert_size_stride(arg156_1, (16, ), (1, ))
    assert_size_stride(arg157_1, (768, 768), (768, 1))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (3072, 768), (768, 1))
    assert_size_stride(arg163_1, (3072, ), (1, ))
    assert_size_stride(arg164_1, (768, 3072), (3072, 1))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (2304, 768), (768, 1))
    assert_size_stride(arg170_1, (2304, ), (1, ))
    assert_size_stride(arg171_1, (16, 16), (16, 1))
    assert_size_stride(arg172_1, (16, ), (1, ))
    assert_size_stride(arg173_1, (16, 16), (16, 1))
    assert_size_stride(arg174_1, (16, ), (1, ))
    assert_size_stride(arg175_1, (768, 768), (768, 1))
    assert_size_stride(arg176_1, (768, ), (1, ))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (3072, 768), (768, 1))
    assert_size_stride(arg181_1, (3072, ), (1, ))
    assert_size_stride(arg182_1, (768, 3072), (3072, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (2304, 768), (768, 1))
    assert_size_stride(arg188_1, (2304, ), (1, ))
    assert_size_stride(arg189_1, (16, 16), (16, 1))
    assert_size_stride(arg190_1, (16, ), (1, ))
    assert_size_stride(arg191_1, (16, 16), (16, 1))
    assert_size_stride(arg192_1, (16, ), (1, ))
    assert_size_stride(arg193_1, (768, 768), (768, 1))
    assert_size_stride(arg194_1, (768, ), (1, ))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (3072, 768), (768, 1))
    assert_size_stride(arg199_1, (3072, ), (1, ))
    assert_size_stride(arg200_1, (768, 3072), (3072, 1))
    assert_size_stride(arg201_1, (768, ), (1, ))
    assert_size_stride(arg202_1, (768, ), (1, ))
    assert_size_stride(arg203_1, (768, ), (1, ))
    assert_size_stride(arg204_1, (768, ), (1, ))
    assert_size_stride(arg205_1, (2304, 768), (768, 1))
    assert_size_stride(arg206_1, (2304, ), (1, ))
    assert_size_stride(arg207_1, (16, 16), (16, 1))
    assert_size_stride(arg208_1, (16, ), (1, ))
    assert_size_stride(arg209_1, (16, 16), (16, 1))
    assert_size_stride(arg210_1, (16, ), (1, ))
    assert_size_stride(arg211_1, (768, 768), (768, 1))
    assert_size_stride(arg212_1, (768, ), (1, ))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (768, ), (1, ))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (3072, 768), (768, 1))
    assert_size_stride(arg217_1, (3072, ), (1, ))
    assert_size_stride(arg218_1, (768, 3072), (3072, 1))
    assert_size_stride(arg219_1, (768, ), (1, ))
    assert_size_stride(arg220_1, (768, ), (1, ))
    assert_size_stride(arg221_1, (768, ), (1, ))
    assert_size_stride(arg222_1, (768, ), (1, ))
    assert_size_stride(arg223_1, (2304, 768), (768, 1))
    assert_size_stride(arg224_1, (2304, ), (1, ))
    assert_size_stride(arg225_1, (16, 16), (16, 1))
    assert_size_stride(arg226_1, (16, ), (1, ))
    assert_size_stride(arg227_1, (16, 16), (16, 1))
    assert_size_stride(arg228_1, (16, ), (1, ))
    assert_size_stride(arg229_1, (768, 768), (768, 1))
    assert_size_stride(arg230_1, (768, ), (1, ))
    assert_size_stride(arg231_1, (768, ), (1, ))
    assert_size_stride(arg232_1, (768, ), (1, ))
    assert_size_stride(arg233_1, (768, ), (1, ))
    assert_size_stride(arg234_1, (3072, 768), (768, 1))
    assert_size_stride(arg235_1, (3072, ), (1, ))
    assert_size_stride(arg236_1, (768, 3072), (3072, 1))
    assert_size_stride(arg237_1, (768, ), (1, ))
    assert_size_stride(arg238_1, (768, ), (1, ))
    assert_size_stride(arg239_1, (768, ), (1, ))
    assert_size_stride(arg240_1, (768, ), (1, ))
    assert_size_stride(arg241_1, (2304, 768), (768, 1))
    assert_size_stride(arg242_1, (2304, ), (1, ))
    assert_size_stride(arg243_1, (16, 16), (16, 1))
    assert_size_stride(arg244_1, (16, ), (1, ))
    assert_size_stride(arg245_1, (16, 16), (16, 1))
    assert_size_stride(arg246_1, (16, ), (1, ))
    assert_size_stride(arg247_1, (768, 768), (768, 1))
    assert_size_stride(arg248_1, (768, ), (1, ))
    assert_size_stride(arg249_1, (768, ), (1, ))
    assert_size_stride(arg250_1, (768, ), (1, ))
    assert_size_stride(arg251_1, (768, ), (1, ))
    assert_size_stride(arg252_1, (3072, 768), (768, 1))
    assert_size_stride(arg253_1, (3072, ), (1, ))
    assert_size_stride(arg254_1, (768, 3072), (3072, 1))
    assert_size_stride(arg255_1, (768, ), (1, ))
    assert_size_stride(arg256_1, (768, ), (1, ))
    assert_size_stride(arg257_1, (768, ), (1, ))
    assert_size_stride(arg258_1, (768, ), (1, ))
    assert_size_stride(arg259_1, (2304, 768), (768, 1))
    assert_size_stride(arg260_1, (2304, ), (1, ))
    assert_size_stride(arg261_1, (16, 16), (16, 1))
    assert_size_stride(arg262_1, (16, ), (1, ))
    assert_size_stride(arg263_1, (16, 16), (16, 1))
    assert_size_stride(arg264_1, (16, ), (1, ))
    assert_size_stride(arg265_1, (768, 768), (768, 1))
    assert_size_stride(arg266_1, (768, ), (1, ))
    assert_size_stride(arg267_1, (768, ), (1, ))
    assert_size_stride(arg268_1, (768, ), (1, ))
    assert_size_stride(arg269_1, (768, ), (1, ))
    assert_size_stride(arg270_1, (3072, 768), (768, 1))
    assert_size_stride(arg271_1, (3072, ), (1, ))
    assert_size_stride(arg272_1, (768, 3072), (3072, 1))
    assert_size_stride(arg273_1, (768, ), (1, ))
    assert_size_stride(arg274_1, (768, ), (1, ))
    assert_size_stride(arg275_1, (768, ), (1, ))
    assert_size_stride(arg276_1, (768, ), (1, ))
    assert_size_stride(arg277_1, (2304, 768), (768, 1))
    assert_size_stride(arg278_1, (2304, ), (1, ))
    assert_size_stride(arg279_1, (16, 16), (16, 1))
    assert_size_stride(arg280_1, (16, ), (1, ))
    assert_size_stride(arg281_1, (16, 16), (16, 1))
    assert_size_stride(arg282_1, (16, ), (1, ))
    assert_size_stride(arg283_1, (768, 768), (768, 1))
    assert_size_stride(arg284_1, (768, ), (1, ))
    assert_size_stride(arg285_1, (768, ), (1, ))
    assert_size_stride(arg286_1, (768, ), (1, ))
    assert_size_stride(arg287_1, (768, ), (1, ))
    assert_size_stride(arg288_1, (3072, 768), (768, 1))
    assert_size_stride(arg289_1, (3072, ), (1, ))
    assert_size_stride(arg290_1, (768, 3072), (3072, 1))
    assert_size_stride(arg291_1, (768, ), (1, ))
    assert_size_stride(arg292_1, (768, ), (1, ))
    assert_size_stride(arg293_1, (768, ), (1, ))
    assert_size_stride(arg294_1, (768, ), (1, ))
    assert_size_stride(arg295_1, (2304, 768), (768, 1))
    assert_size_stride(arg296_1, (2304, ), (1, ))
    assert_size_stride(arg297_1, (16, 16), (16, 1))
    assert_size_stride(arg298_1, (16, ), (1, ))
    assert_size_stride(arg299_1, (16, 16), (16, 1))
    assert_size_stride(arg300_1, (16, ), (1, ))
    assert_size_stride(arg301_1, (768, 768), (768, 1))
    assert_size_stride(arg302_1, (768, ), (1, ))
    assert_size_stride(arg303_1, (768, ), (1, ))
    assert_size_stride(arg304_1, (768, ), (1, ))
    assert_size_stride(arg305_1, (768, ), (1, ))
    assert_size_stride(arg306_1, (3072, 768), (768, 1))
    assert_size_stride(arg307_1, (3072, ), (1, ))
    assert_size_stride(arg308_1, (768, 3072), (3072, 1))
    assert_size_stride(arg309_1, (768, ), (1, ))
    assert_size_stride(arg310_1, (768, ), (1, ))
    assert_size_stride(arg311_1, (768, ), (1, ))
    assert_size_stride(arg312_1, (768, ), (1, ))
    assert_size_stride(arg313_1, (2304, 768), (768, 1))
    assert_size_stride(arg314_1, (2304, ), (1, ))
    assert_size_stride(arg315_1, (16, 16), (16, 1))
    assert_size_stride(arg316_1, (16, ), (1, ))
    assert_size_stride(arg317_1, (16, 16), (16, 1))
    assert_size_stride(arg318_1, (16, ), (1, ))
    assert_size_stride(arg319_1, (768, 768), (768, 1))
    assert_size_stride(arg320_1, (768, ), (1, ))
    assert_size_stride(arg321_1, (768, ), (1, ))
    assert_size_stride(arg322_1, (768, ), (1, ))
    assert_size_stride(arg323_1, (768, ), (1, ))
    assert_size_stride(arg324_1, (3072, 768), (768, 1))
    assert_size_stride(arg325_1, (3072, ), (1, ))
    assert_size_stride(arg326_1, (768, 3072), (3072, 1))
    assert_size_stride(arg327_1, (768, ), (1, ))
    assert_size_stride(arg328_1, (768, ), (1, ))
    assert_size_stride(arg329_1, (768, ), (1, ))
    assert_size_stride(arg330_1, (768, ), (1, ))
    assert_size_stride(arg331_1, (2304, 768), (768, 1))
    assert_size_stride(arg332_1, (2304, ), (1, ))
    assert_size_stride(arg333_1, (16, 16), (16, 1))
    assert_size_stride(arg334_1, (16, ), (1, ))
    assert_size_stride(arg335_1, (16, 16), (16, 1))
    assert_size_stride(arg336_1, (16, ), (1, ))
    assert_size_stride(arg337_1, (768, 768), (768, 1))
    assert_size_stride(arg338_1, (768, ), (1, ))
    assert_size_stride(arg339_1, (768, ), (1, ))
    assert_size_stride(arg340_1, (768, ), (1, ))
    assert_size_stride(arg341_1, (768, ), (1, ))
    assert_size_stride(arg342_1, (3072, 768), (768, 1))
    assert_size_stride(arg343_1, (3072, ), (1, ))
    assert_size_stride(arg344_1, (768, 3072), (3072, 1))
    assert_size_stride(arg345_1, (768, ), (1, ))
    assert_size_stride(arg346_1, (768, ), (1, ))
    assert_size_stride(arg347_1, (768, ), (1, ))
    assert_size_stride(arg348_1, (768, ), (1, ))
    assert_size_stride(arg349_1, (2304, 768), (768, 1))
    assert_size_stride(arg350_1, (2304, ), (1, ))
    assert_size_stride(arg351_1, (16, 16), (16, 1))
    assert_size_stride(arg352_1, (16, ), (1, ))
    assert_size_stride(arg353_1, (16, 16), (16, 1))
    assert_size_stride(arg354_1, (16, ), (1, ))
    assert_size_stride(arg355_1, (768, 768), (768, 1))
    assert_size_stride(arg356_1, (768, ), (1, ))
    assert_size_stride(arg357_1, (768, ), (1, ))
    assert_size_stride(arg358_1, (768, ), (1, ))
    assert_size_stride(arg359_1, (768, ), (1, ))
    assert_size_stride(arg360_1, (3072, 768), (768, 1))
    assert_size_stride(arg361_1, (3072, ), (1, ))
    assert_size_stride(arg362_1, (768, 3072), (3072, 1))
    assert_size_stride(arg363_1, (768, ), (1, ))
    assert_size_stride(arg364_1, (768, ), (1, ))
    assert_size_stride(arg365_1, (768, ), (1, ))
    assert_size_stride(arg366_1, (768, ), (1, ))
    assert_size_stride(arg367_1, (2304, 768), (768, 1))
    assert_size_stride(arg368_1, (2304, ), (1, ))
    assert_size_stride(arg369_1, (16, 16), (16, 1))
    assert_size_stride(arg370_1, (16, ), (1, ))
    assert_size_stride(arg371_1, (16, 16), (16, 1))
    assert_size_stride(arg372_1, (16, ), (1, ))
    assert_size_stride(arg373_1, (768, 768), (768, 1))
    assert_size_stride(arg374_1, (768, ), (1, ))
    assert_size_stride(arg375_1, (768, ), (1, ))
    assert_size_stride(arg376_1, (768, ), (1, ))
    assert_size_stride(arg377_1, (768, ), (1, ))
    assert_size_stride(arg378_1, (3072, 768), (768, 1))
    assert_size_stride(arg379_1, (3072, ), (1, ))
    assert_size_stride(arg380_1, (768, 3072), (3072, 1))
    assert_size_stride(arg381_1, (768, ), (1, ))
    assert_size_stride(arg382_1, (768, ), (1, ))
    assert_size_stride(arg383_1, (768, ), (1, ))
    assert_size_stride(arg384_1, (768, ), (1, ))
    assert_size_stride(arg385_1, (2304, 768), (768, 1))
    assert_size_stride(arg386_1, (2304, ), (1, ))
    assert_size_stride(arg387_1, (16, 16), (16, 1))
    assert_size_stride(arg388_1, (16, ), (1, ))
    assert_size_stride(arg389_1, (16, 16), (16, 1))
    assert_size_stride(arg390_1, (16, ), (1, ))
    assert_size_stride(arg391_1, (768, 768), (768, 1))
    assert_size_stride(arg392_1, (768, ), (1, ))
    assert_size_stride(arg393_1, (768, ), (1, ))
    assert_size_stride(arg394_1, (768, ), (1, ))
    assert_size_stride(arg395_1, (768, ), (1, ))
    assert_size_stride(arg396_1, (3072, 768), (768, 1))
    assert_size_stride(arg397_1, (3072, ), (1, ))
    assert_size_stride(arg398_1, (768, 3072), (3072, 1))
    assert_size_stride(arg399_1, (768, ), (1, ))
    assert_size_stride(arg400_1, (768, ), (1, ))
    assert_size_stride(arg401_1, (768, ), (1, ))
    assert_size_stride(arg402_1, (768, ), (1, ))
    assert_size_stride(arg403_1, (2304, 768), (768, 1))
    assert_size_stride(arg404_1, (2304, ), (1, ))
    assert_size_stride(arg405_1, (16, 16), (16, 1))
    assert_size_stride(arg406_1, (16, ), (1, ))
    assert_size_stride(arg407_1, (16, 16), (16, 1))
    assert_size_stride(arg408_1, (16, ), (1, ))
    assert_size_stride(arg409_1, (768, 768), (768, 1))
    assert_size_stride(arg410_1, (768, ), (1, ))
    assert_size_stride(arg411_1, (768, ), (1, ))
    assert_size_stride(arg412_1, (768, ), (1, ))
    assert_size_stride(arg413_1, (768, ), (1, ))
    assert_size_stride(arg414_1, (3072, 768), (768, 1))
    assert_size_stride(arg415_1, (3072, ), (1, ))
    assert_size_stride(arg416_1, (768, 3072), (3072, 1))
    assert_size_stride(arg417_1, (768, ), (1, ))
    assert_size_stride(arg418_1, (768, ), (1, ))
    assert_size_stride(arg419_1, (768, ), (1, ))
    assert_size_stride(arg420_1, (768, ), (1, ))
    assert_size_stride(arg421_1, (2304, 768), (768, 1))
    assert_size_stride(arg422_1, (2304, ), (1, ))
    assert_size_stride(arg423_1, (16, 16), (16, 1))
    assert_size_stride(arg424_1, (16, ), (1, ))
    assert_size_stride(arg425_1, (16, 16), (16, 1))
    assert_size_stride(arg426_1, (16, ), (1, ))
    assert_size_stride(arg427_1, (768, 768), (768, 1))
    assert_size_stride(arg428_1, (768, ), (1, ))
    assert_size_stride(arg429_1, (768, ), (1, ))
    assert_size_stride(arg430_1, (768, ), (1, ))
    assert_size_stride(arg431_1, (768, ), (1, ))
    assert_size_stride(arg432_1, (3072, 768), (768, 1))
    assert_size_stride(arg433_1, (3072, ), (1, ))
    assert_size_stride(arg434_1, (768, 3072), (3072, 1))
    assert_size_stride(arg435_1, (768, ), (1, ))
    assert_size_stride(arg436_1, (768, ), (1, ))
    assert_size_stride(arg437_1, (768, ), (1, ))
    assert_size_stride(arg438_1, (768, ), (1, ))
    assert_size_stride(arg439_1, (2304, 768), (768, 1))
    assert_size_stride(arg440_1, (2304, ), (1, ))
    assert_size_stride(arg441_1, (16, 16), (16, 1))
    assert_size_stride(arg442_1, (16, ), (1, ))
    assert_size_stride(arg443_1, (16, 16), (16, 1))
    assert_size_stride(arg444_1, (16, ), (1, ))
    assert_size_stride(arg445_1, (768, 768), (768, 1))
    assert_size_stride(arg446_1, (768, ), (1, ))
    assert_size_stride(arg447_1, (768, ), (1, ))
    assert_size_stride(arg448_1, (768, ), (1, ))
    assert_size_stride(arg449_1, (768, ), (1, ))
    assert_size_stride(arg450_1, (3072, 768), (768, 1))
    assert_size_stride(arg451_1, (3072, ), (1, ))
    assert_size_stride(arg452_1, (768, 3072), (3072, 1))
    assert_size_stride(arg453_1, (768, ), (1, ))
    assert_size_stride(arg454_1, (768, ), (1, ))
    assert_size_stride(arg455_1, (768, ), (1, ))
    assert_size_stride(arg456_1, (768, ), (1, ))
    assert_size_stride(arg457_1, (2304, 768), (768, 1))
    assert_size_stride(arg458_1, (2304, ), (1, ))
    assert_size_stride(arg459_1, (16, 16), (16, 1))
    assert_size_stride(arg460_1, (16, ), (1, ))
    assert_size_stride(arg461_1, (16, 16), (16, 1))
    assert_size_stride(arg462_1, (16, ), (1, ))
    assert_size_stride(arg463_1, (768, 768), (768, 1))
    assert_size_stride(arg464_1, (768, ), (1, ))
    assert_size_stride(arg465_1, (768, ), (1, ))
    assert_size_stride(arg466_1, (768, ), (1, ))
    assert_size_stride(arg467_1, (768, ), (1, ))
    assert_size_stride(arg468_1, (3072, 768), (768, 1))
    assert_size_stride(arg469_1, (3072, ), (1, ))
    assert_size_stride(arg470_1, (768, 3072), (3072, 1))
    assert_size_stride(arg471_1, (768, ), (1, ))
    assert_size_stride(arg472_1, (768, ), (1, ))
    assert_size_stride(arg473_1, (768, ), (1, ))
    assert_size_stride(arg474_1, (768, ), (1, ))
    assert_size_stride(arg475_1, (2304, 768), (768, 1))
    assert_size_stride(arg476_1, (2304, ), (1, ))
    assert_size_stride(arg477_1, (16, 16), (16, 1))
    assert_size_stride(arg478_1, (16, ), (1, ))
    assert_size_stride(arg479_1, (16, 16), (16, 1))
    assert_size_stride(arg480_1, (16, ), (1, ))
    assert_size_stride(arg481_1, (768, 768), (768, 1))
    assert_size_stride(arg482_1, (768, ), (1, ))
    assert_size_stride(arg483_1, (768, ), (1, ))
    assert_size_stride(arg484_1, (768, ), (1, ))
    assert_size_stride(arg485_1, (768, ), (1, ))
    assert_size_stride(arg486_1, (3072, 768), (768, 1))
    assert_size_stride(arg487_1, (3072, ), (1, ))
    assert_size_stride(arg488_1, (768, 3072), (3072, 1))
    assert_size_stride(arg489_1, (768, ), (1, ))
    assert_size_stride(arg490_1, (768, ), (1, ))
    assert_size_stride(arg491_1, (768, ), (1, ))
    assert_size_stride(arg492_1, (768, ), (1, ))
    assert_size_stride(arg493_1, (2304, 768), (768, 1))
    assert_size_stride(arg494_1, (2304, ), (1, ))
    assert_size_stride(arg495_1, (16, 16), (16, 1))
    assert_size_stride(arg496_1, (16, ), (1, ))
    assert_size_stride(arg497_1, (16, 16), (16, 1))
    assert_size_stride(arg498_1, (16, ), (1, ))
    assert_size_stride(arg499_1, (768, 768), (768, 1))
    assert_size_stride(arg500_1, (768, ), (1, ))
    assert_size_stride(arg501_1, (768, ), (1, ))
    assert_size_stride(arg502_1, (768, ), (1, ))
    assert_size_stride(arg503_1, (768, ), (1, ))
    assert_size_stride(arg504_1, (3072, 768), (768, 1))
    assert_size_stride(arg505_1, (3072, ), (1, ))
    assert_size_stride(arg506_1, (768, 3072), (3072, 1))
    assert_size_stride(arg507_1, (768, ), (1, ))
    assert_size_stride(arg508_1, (768, ), (1, ))
    assert_size_stride(arg509_1, (768, ), (1, ))
    assert_size_stride(arg510_1, (768, ), (1, ))
    assert_size_stride(arg511_1, (2304, 768), (768, 1))
    assert_size_stride(arg512_1, (2304, ), (1, ))
    assert_size_stride(arg513_1, (16, 16), (16, 1))
    assert_size_stride(arg514_1, (16, ), (1, ))
    assert_size_stride(arg515_1, (16, 16), (16, 1))
    assert_size_stride(arg516_1, (16, ), (1, ))
    assert_size_stride(arg517_1, (768, 768), (768, 1))
    assert_size_stride(arg518_1, (768, ), (1, ))
    assert_size_stride(arg519_1, (768, ), (1, ))
    assert_size_stride(arg520_1, (768, ), (1, ))
    assert_size_stride(arg521_1, (768, ), (1, ))
    assert_size_stride(arg522_1, (3072, 768), (768, 1))
    assert_size_stride(arg523_1, (3072, ), (1, ))
    assert_size_stride(arg524_1, (768, 3072), (3072, 1))
    assert_size_stride(arg525_1, (768, ), (1, ))
    assert_size_stride(arg526_1, (768, ), (1, ))
    assert_size_stride(arg527_1, (768, ), (1, ))
    assert_size_stride(arg528_1, (768, ), (1, ))
    assert_size_stride(arg529_1, (2304, 768), (768, 1))
    assert_size_stride(arg530_1, (2304, ), (1, ))
    assert_size_stride(arg531_1, (16, 16), (16, 1))
    assert_size_stride(arg532_1, (16, ), (1, ))
    assert_size_stride(arg533_1, (16, 16), (16, 1))
    assert_size_stride(arg534_1, (16, ), (1, ))
    assert_size_stride(arg535_1, (768, 768), (768, 1))
    assert_size_stride(arg536_1, (768, ), (1, ))
    assert_size_stride(arg537_1, (768, ), (1, ))
    assert_size_stride(arg538_1, (768, ), (1, ))
    assert_size_stride(arg539_1, (768, ), (1, ))
    assert_size_stride(arg540_1, (3072, 768), (768, 1))
    assert_size_stride(arg541_1, (3072, ), (1, ))
    assert_size_stride(arg542_1, (768, 3072), (3072, 1))
    assert_size_stride(arg543_1, (768, ), (1, ))
    assert_size_stride(arg544_1, (768, ), (1, ))
    assert_size_stride(arg545_1, (768, ), (1, ))
    assert_size_stride(arg546_1, (768, ), (1, ))
    assert_size_stride(arg547_1, (2304, 768), (768, 1))
    assert_size_stride(arg548_1, (2304, ), (1, ))
    assert_size_stride(arg549_1, (16, 16), (16, 1))
    assert_size_stride(arg550_1, (16, ), (1, ))
    assert_size_stride(arg551_1, (16, 16), (16, 1))
    assert_size_stride(arg552_1, (16, ), (1, ))
    assert_size_stride(arg553_1, (768, 768), (768, 1))
    assert_size_stride(arg554_1, (768, ), (1, ))
    assert_size_stride(arg555_1, (768, ), (1, ))
    assert_size_stride(arg556_1, (768, ), (1, ))
    assert_size_stride(arg557_1, (768, ), (1, ))
    assert_size_stride(arg558_1, (3072, 768), (768, 1))
    assert_size_stride(arg559_1, (3072, ), (1, ))
    assert_size_stride(arg560_1, (768, 3072), (3072, 1))
    assert_size_stride(arg561_1, (768, ), (1, ))
    assert_size_stride(arg562_1, (768, ), (1, ))
    assert_size_stride(arg563_1, (768, ), (1, ))
    assert_size_stride(arg564_1, (768, ), (1, ))
    assert_size_stride(arg565_1, (2304, 768), (768, 1))
    assert_size_stride(arg566_1, (2304, ), (1, ))
    assert_size_stride(arg567_1, (16, 16), (16, 1))
    assert_size_stride(arg568_1, (16, ), (1, ))
    assert_size_stride(arg569_1, (16, 16), (16, 1))
    assert_size_stride(arg570_1, (16, ), (1, ))
    assert_size_stride(arg571_1, (768, 768), (768, 1))
    assert_size_stride(arg572_1, (768, ), (1, ))
    assert_size_stride(arg573_1, (768, ), (1, ))
    assert_size_stride(arg574_1, (768, ), (1, ))
    assert_size_stride(arg575_1, (768, ), (1, ))
    assert_size_stride(arg576_1, (3072, 768), (768, 1))
    assert_size_stride(arg577_1, (3072, ), (1, ))
    assert_size_stride(arg578_1, (768, 3072), (3072, 1))
    assert_size_stride(arg579_1, (768, ), (1, ))
    assert_size_stride(arg580_1, (768, ), (1, ))
    assert_size_stride(arg581_1, (768, ), (1, ))
    assert_size_stride(arg582_1, (768, ), (1, ))
    assert_size_stride(arg583_1, (2304, 768), (768, 1))
    assert_size_stride(arg584_1, (2304, ), (1, ))
    assert_size_stride(arg585_1, (16, 16), (16, 1))
    assert_size_stride(arg586_1, (16, ), (1, ))
    assert_size_stride(arg587_1, (16, 16), (16, 1))
    assert_size_stride(arg588_1, (16, ), (1, ))
    assert_size_stride(arg589_1, (768, 768), (768, 1))
    assert_size_stride(arg590_1, (768, ), (1, ))
    assert_size_stride(arg591_1, (768, ), (1, ))
    assert_size_stride(arg592_1, (768, ), (1, ))
    assert_size_stride(arg593_1, (768, ), (1, ))
    assert_size_stride(arg594_1, (3072, 768), (768, 1))
    assert_size_stride(arg595_1, (3072, ), (1, ))
    assert_size_stride(arg596_1, (768, 3072), (3072, 1))
    assert_size_stride(arg597_1, (768, ), (1, ))
    assert_size_stride(arg598_1, (768, ), (1, ))
    assert_size_stride(arg599_1, (768, ), (1, ))
    assert_size_stride(arg600_1, (768, ), (1, ))
    assert_size_stride(arg601_1, (2304, 768), (768, 1))
    assert_size_stride(arg602_1, (2304, ), (1, ))
    assert_size_stride(arg603_1, (16, 16), (16, 1))
    assert_size_stride(arg604_1, (16, ), (1, ))
    assert_size_stride(arg605_1, (16, 16), (16, 1))
    assert_size_stride(arg606_1, (16, ), (1, ))
    assert_size_stride(arg607_1, (768, 768), (768, 1))
    assert_size_stride(arg608_1, (768, ), (1, ))
    assert_size_stride(arg609_1, (768, ), (1, ))
    assert_size_stride(arg610_1, (768, ), (1, ))
    assert_size_stride(arg611_1, (768, ), (1, ))
    assert_size_stride(arg612_1, (3072, 768), (768, 1))
    assert_size_stride(arg613_1, (3072, ), (1, ))
    assert_size_stride(arg614_1, (768, 3072), (3072, 1))
    assert_size_stride(arg615_1, (768, ), (1, ))
    assert_size_stride(arg616_1, (768, ), (1, ))
    assert_size_stride(arg617_1, (768, ), (1, ))
    assert_size_stride(arg618_1, (768, ), (1, ))
    assert_size_stride(arg619_1, (2304, 768), (768, 1))
    assert_size_stride(arg620_1, (2304, ), (1, ))
    assert_size_stride(arg621_1, (16, 16), (16, 1))
    assert_size_stride(arg622_1, (16, ), (1, ))
    assert_size_stride(arg623_1, (16, 16), (16, 1))
    assert_size_stride(arg624_1, (16, ), (1, ))
    assert_size_stride(arg625_1, (768, 768), (768, 1))
    assert_size_stride(arg626_1, (768, ), (1, ))
    assert_size_stride(arg627_1, (768, ), (1, ))
    assert_size_stride(arg628_1, (768, ), (1, ))
    assert_size_stride(arg629_1, (768, ), (1, ))
    assert_size_stride(arg630_1, (3072, 768), (768, 1))
    assert_size_stride(arg631_1, (3072, ), (1, ))
    assert_size_stride(arg632_1, (768, 3072), (3072, 1))
    assert_size_stride(arg633_1, (768, ), (1, ))
    assert_size_stride(arg634_1, (768, ), (1, ))
    assert_size_stride(arg635_1, (768, ), (1, ))
    assert_size_stride(arg636_1, (768, ), (1, ))
    assert_size_stride(arg637_1, (2304, 768), (768, 1))
    assert_size_stride(arg638_1, (2304, ), (1, ))
    assert_size_stride(arg639_1, (16, 16), (16, 1))
    assert_size_stride(arg640_1, (16, ), (1, ))
    assert_size_stride(arg641_1, (16, 16), (16, 1))
    assert_size_stride(arg642_1, (16, ), (1, ))
    assert_size_stride(arg643_1, (768, 768), (768, 1))
    assert_size_stride(arg644_1, (768, ), (1, ))
    assert_size_stride(arg645_1, (768, ), (1, ))
    assert_size_stride(arg646_1, (768, ), (1, ))
    assert_size_stride(arg647_1, (768, ), (1, ))
    assert_size_stride(arg648_1, (3072, 768), (768, 1))
    assert_size_stride(arg649_1, (3072, ), (1, ))
    assert_size_stride(arg650_1, (768, 3072), (3072, 1))
    assert_size_stride(arg651_1, (768, ), (1, ))
    assert_size_stride(arg652_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg653_1, (768, ), (1, ))
    assert_size_stride(arg654_1, (768, ), (1, ))
    assert_size_stride(arg655_1, (768, ), (1, ))
    assert_size_stride(arg656_1, (768, 768), (768, 1))
    assert_size_stride(arg657_1, (768, ), (1, ))
    assert_size_stride(arg658_1, (768, 768), (768, 1))
    assert_size_stride(arg659_1, (768, ), (1, ))
    assert_size_stride(arg660_1, (768, 768), (768, 1))
    assert_size_stride(arg661_1, (768, ), (1, ))
    assert_size_stride(arg662_1, (768, 768), (768, 1))
    assert_size_stride(arg663_1, (768, ), (1, ))
    assert_size_stride(arg664_1, (768, ), (1, ))
    assert_size_stride(arg665_1, (768, ), (1, ))
    assert_size_stride(arg666_1, (768, ), (1, ))
    assert_size_stride(arg667_1, (3072, 768), (768, 1))
    assert_size_stride(arg668_1, (3072, ), (1, ))
    assert_size_stride(arg669_1, (768, 3072), (3072, 1))
    assert_size_stride(arg670_1, (768, ), (1, ))
    assert_size_stride(arg671_1, (768, ), (1, ))
    assert_size_stride(arg672_1, (768, ), (1, ))
    assert_size_stride(arg673_1, (768, ), (1, ))
    assert_size_stride(arg674_1, (768, 768), (768, 1))
    assert_size_stride(arg675_1, (768, ), (1, ))
    assert_size_stride(arg676_1, (768, 768), (768, 1))
    assert_size_stride(arg677_1, (768, ), (1, ))
    assert_size_stride(arg678_1, (768, 768), (768, 1))
    assert_size_stride(arg679_1, (768, ), (1, ))
    assert_size_stride(arg680_1, (768, 768), (768, 1))
    assert_size_stride(arg681_1, (768, ), (1, ))
    assert_size_stride(arg682_1, (768, ), (1, ))
    assert_size_stride(arg683_1, (768, ), (1, ))
    assert_size_stride(arg684_1, (768, ), (1, ))
    assert_size_stride(arg685_1, (3072, 768), (768, 1))
    assert_size_stride(arg686_1, (3072, ), (1, ))
    assert_size_stride(arg687_1, (768, 3072), (3072, 1))
    assert_size_stride(arg688_1, (768, ), (1, ))
    assert_size_stride(arg689_1, (768, ), (1, ))
    assert_size_stride(arg690_1, (768, ), (1, ))
    assert_size_stride(arg691_1, (1000, 768), (768, 1))
    assert_size_stride(arg692_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_379], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 24, 24), (442368, 576, 24, 1))
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((8, 576, 1, 6), (3456, 6, 27648, 1), torch.float32)
        buf2 = empty_strided_cuda((8, 576, 1, 6), (3456, 6, 27648, 1), torch.float32)
        buf3 = empty_strided_cuda((8, 576, 1, 6), (3456, 6, 27648, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_381, layer_norm_77], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_0.run(buf0, arg2_1, arg3_1, buf1, buf2, buf3, 27648, 128, grid=grid(27648), stream=stream0)
        buf4 = empty_strided_cuda((8, 576, 1), (576, 1, 4608), torch.float32)
        buf5 = empty_strided_cuda((8, 576, 1), (576, 1, 4608), torch.float32)
        # Topologically Sorted Source Nodes: [x_381, layer_norm_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 4608, 6, grid=grid(4608), stream=stream0)
        del buf1
        del buf2
        del buf3
        buf7 = empty_strided_cuda((8, 576, 768), (442368, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_381, layer_norm_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_2.run(buf0, arg2_1, arg3_1, buf4, buf5, arg5_1, arg6_1, buf7, 4608, 768, grid=grid(4608, 768), stream=stream0)
        del arg5_1
        del arg6_1
        del buf4
        del buf5
        buf8 = empty_strided_cuda((4608, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (4608, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 2304), (1, 768), 0), out=buf8)
        del arg7_1
        buf9 = reinterpret_tensor(buf7, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [q_38, attn_180], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf8, arg8_1, buf9, 3538944, grid=grid(3538944), stream=stream0)
        buf10 = empty_strided_cuda((8, 16, 48, 576), (442368, 27648, 576, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_180], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf8, arg8_1, buf10, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf11 = empty_strided_cuda((128, 576, 576), (331776, 576, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_180], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf10, (128, 48, 576), (27648, 576, 1), 0), out=buf11)
        buf12 = empty_strided_cuda((8, 576, 576, 16), (5308416, 9216, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_230], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf11, buf12, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf13 = reinterpret_tensor(buf11, (2654208, 16), (16, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [linear_230], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg9_1, (16, 16), (1, 16), 0), out=buf13)
        del arg9_1
        buf14 = empty_strided_cuda((8, 16, 576, 1), (9216, 1, 16, 73728), torch.float32)
        buf15 = empty_strided_cuda((8, 16, 576, 1), (9216, 1, 16, 73728), torch.float32)
        # Topologically Sorted Source Nodes: [attn_182], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf13, arg10_1, buf14, buf15, 73728, 576, grid=grid(73728), stream=stream0)
        buf16 = reinterpret_tensor(buf13, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [linear_231], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf16, arg10_1, buf14, buf15, 42467328, grid=grid(42467328), stream=stream0)
        del arg10_1
        buf17 = reinterpret_tensor(buf12, (2654208, 16), (16, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [linear_231], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg11_1, (16, 16), (1, 16), 0), out=buf17)
        del arg11_1
        buf18 = reinterpret_tensor(buf16, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [matmul_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf17, arg12_1, buf18, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg12_1
        buf19 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [matmul_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf8, arg8_1, buf19, 3538944, grid=grid(3538944), stream=stream0)
        del arg8_1
        buf20 = reinterpret_tensor(buf10, (128, 576, 48), (27648, 48, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [matmul_73], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf18, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf19, (128, 576, 48), (27648, 48, 1), 0), out=buf20)
        buf21 = reinterpret_tensor(buf19, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_383], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf20, buf21, 3538944, grid=grid(3538944), stream=stream0)
        buf22 = reinterpret_tensor(buf20, (4608, 768), (768, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (4608, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), out=buf22)
        del arg13_1
        buf23 = reinterpret_tensor(buf22, (8, 576, 768), (442368, 768, 1), 0); del buf22  # reuse
        buf27 = reinterpret_tensor(buf21, (8, 576, 768), (442368, 768, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_381, mul_113, x_386, layer_norm_78], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_11.run(buf23, buf0, arg2_1, arg3_1, arg4_1, arg14_1, arg16_1, arg17_1, buf27, 4608, 768, grid=grid(4608), stream=stream0)
        del arg14_1
        del arg16_1
        del arg17_1
        del arg2_1
        del arg3_1
        del arg4_1
        buf28 = empty_strided_cuda((4608, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (4608, 768), (768, 1), 0), reinterpret_tensor(arg18_1, (768, 3072), (1, 768), 0), out=buf28)
        del arg18_1
        buf29 = reinterpret_tensor(buf28, (8, 576, 3072), (1769472, 3072, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_388], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf29, arg19_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg19_1
        buf30 = reinterpret_tensor(buf27, (4608, 768), (768, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg20_1, (3072, 768), (1, 3072), 0), out=buf30)
        del arg20_1
        buf34 = reinterpret_tensor(buf0, (8, 576, 768), (442368, 768, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [mul_114, x_392, layer_norm_79], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf23, arg15_1, buf30, arg21_1, arg23_1, arg24_1, buf34, 4608, 768, grid=grid(4608), stream=stream0)
        del arg23_1
        del arg24_1
        buf35 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (4608, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 2304), (1, 768), 0), out=buf35)
        del arg25_1
        buf36 = reinterpret_tensor(buf34, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [q_39, attn_185], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf35, arg26_1, buf36, 3538944, grid=grid(3538944), stream=stream0)
        buf37 = empty_strided_cuda((8, 16, 48, 576), (442368, 27648, 576, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_185], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf35, arg26_1, buf37, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf38 = reinterpret_tensor(buf18, (128, 576, 576), (331776, 576, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [attn_185], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf37, (128, 48, 576), (27648, 576, 1), 0), out=buf38)
        buf39 = reinterpret_tensor(buf17, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [linear_236], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf38, buf39, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf40 = reinterpret_tensor(buf38, (2654208, 16), (16, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [linear_236], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg27_1, (16, 16), (1, 16), 0), out=buf40)
        del arg27_1
        buf41 = buf15; del buf15  # reuse
        buf42 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [attn_187], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf40, arg28_1, buf41, buf42, 73728, 576, grid=grid(73728), stream=stream0)
        buf43 = reinterpret_tensor(buf40, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [linear_237], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf43, arg28_1, buf41, buf42, 42467328, grid=grid(42467328), stream=stream0)
        del arg28_1
        buf44 = reinterpret_tensor(buf39, (2654208, 16), (16, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [linear_237], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg29_1, (16, 16), (1, 16), 0), out=buf44)
        del arg29_1
        buf45 = reinterpret_tensor(buf43, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [matmul_75], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf44, arg30_1, buf45, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg30_1
        buf46 = reinterpret_tensor(buf37, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [matmul_75], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf35, arg26_1, buf46, 3538944, grid=grid(3538944), stream=stream0)
        del arg26_1
        buf47 = reinterpret_tensor(buf36, (128, 576, 48), (27648, 48, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [matmul_75], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf45, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf46, (128, 576, 48), (27648, 48, 1), 0), out=buf47)
        buf48 = reinterpret_tensor(buf46, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_393], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf47, buf48, 3538944, grid=grid(3538944), stream=stream0)
        buf49 = reinterpret_tensor(buf47, (4608, 768), (768, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf48, (4608, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 768), (1, 768), 0), out=buf49)
        del arg31_1
        buf50 = reinterpret_tensor(buf49, (8, 576, 768), (442368, 768, 1), 0); del buf49  # reuse
        buf54 = reinterpret_tensor(buf48, (8, 576, 768), (442368, 768, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [mul_114, x_392, mul_116, x_396, layer_norm_80], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf50, buf23, arg15_1, buf30, arg21_1, arg22_1, arg32_1, arg34_1, arg35_1, buf54, 4608, 768, grid=grid(4608), stream=stream0)
        del arg15_1
        del arg21_1
        del arg22_1
        del arg32_1
        del arg34_1
        del arg35_1
        buf55 = reinterpret_tensor(buf29, (4608, 3072), (3072, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf54, (4608, 768), (768, 1), 0), reinterpret_tensor(arg36_1, (768, 3072), (1, 768), 0), out=buf55)
        del arg36_1
        buf56 = reinterpret_tensor(buf55, (8, 576, 3072), (1769472, 3072, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [x_398], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf56, arg37_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg37_1
        buf57 = reinterpret_tensor(buf54, (4608, 768), (768, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg38_1, (3072, 768), (1, 3072), 0), out=buf57)
        del arg38_1
        buf61 = reinterpret_tensor(buf30, (8, 576, 768), (442368, 768, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [mul_117, x_402, layer_norm_81], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf50, arg33_1, buf57, arg39_1, arg41_1, arg42_1, buf61, 4608, 768, grid=grid(4608), stream=stream0)
        del arg41_1
        del arg42_1
        buf62 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (4608, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 2304), (1, 768), 0), out=buf62)
        del arg43_1
        buf63 = reinterpret_tensor(buf61, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [q_40, attn_190], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf62, arg44_1, buf63, 3538944, grid=grid(3538944), stream=stream0)
        buf64 = reinterpret_tensor(buf23, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [attn_190], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf62, arg44_1, buf64, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf65 = reinterpret_tensor(buf45, (128, 576, 576), (331776, 576, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [attn_190], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf63, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf64, (128, 48, 576), (27648, 576, 1), 0), out=buf65)
        buf66 = reinterpret_tensor(buf44, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [linear_242], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf65, buf66, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf67 = reinterpret_tensor(buf65, (2654208, 16), (16, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [linear_242], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg45_1, (16, 16), (1, 16), 0), out=buf67)
        del arg45_1
        buf68 = buf42; del buf42  # reuse
        buf69 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [attn_192], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf67, arg46_1, buf68, buf69, 73728, 576, grid=grid(73728), stream=stream0)
        buf70 = reinterpret_tensor(buf67, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [linear_243], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf70, arg46_1, buf68, buf69, 42467328, grid=grid(42467328), stream=stream0)
        del arg46_1
        buf71 = reinterpret_tensor(buf66, (2654208, 16), (16, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [linear_243], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg47_1, (16, 16), (1, 16), 0), out=buf71)
        del arg47_1
        buf72 = reinterpret_tensor(buf70, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [matmul_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf71, arg48_1, buf72, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg48_1
        buf73 = reinterpret_tensor(buf64, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [matmul_77], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf62, arg44_1, buf73, 3538944, grid=grid(3538944), stream=stream0)
        del arg44_1
        buf74 = reinterpret_tensor(buf63, (128, 576, 48), (27648, 48, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [matmul_77], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf72, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf73, (128, 576, 48), (27648, 48, 1), 0), out=buf74)
        buf75 = reinterpret_tensor(buf73, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_403], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf74, buf75, 3538944, grid=grid(3538944), stream=stream0)
        buf76 = reinterpret_tensor(buf74, (4608, 768), (768, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (4608, 768), (768, 1), 0), reinterpret_tensor(arg49_1, (768, 768), (1, 768), 0), out=buf76)
        del arg49_1
        buf77 = reinterpret_tensor(buf76, (8, 576, 768), (442368, 768, 1), 0); del buf76  # reuse
        buf81 = reinterpret_tensor(buf75, (8, 576, 768), (442368, 768, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [mul_117, x_402, mul_119, x_406, layer_norm_82], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf77, buf50, arg33_1, buf57, arg39_1, arg40_1, arg50_1, arg52_1, arg53_1, buf81, 4608, 768, grid=grid(4608), stream=stream0)
        del arg33_1
        del arg39_1
        del arg40_1
        del arg50_1
        del arg52_1
        del arg53_1
        buf82 = reinterpret_tensor(buf56, (4608, 3072), (3072, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf81, (4608, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 3072), (1, 768), 0), out=buf82)
        del arg54_1
        buf83 = reinterpret_tensor(buf82, (8, 576, 3072), (1769472, 3072, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [x_408], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf83, arg55_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg55_1
        buf84 = reinterpret_tensor(buf81, (4608, 768), (768, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf83, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg56_1, (3072, 768), (1, 3072), 0), out=buf84)
        del arg56_1
        buf88 = reinterpret_tensor(buf57, (8, 576, 768), (442368, 768, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [mul_120, x_412, layer_norm_83], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf77, arg51_1, buf84, arg57_1, arg59_1, arg60_1, buf88, 4608, 768, grid=grid(4608), stream=stream0)
        del arg59_1
        del arg60_1
        buf89 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf88, (4608, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 2304), (1, 768), 0), out=buf89)
        del arg61_1
        buf90 = reinterpret_tensor(buf88, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [q_41, attn_195], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf89, arg62_1, buf90, 3538944, grid=grid(3538944), stream=stream0)
        buf91 = reinterpret_tensor(buf50, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [attn_195], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf89, arg62_1, buf91, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf92 = reinterpret_tensor(buf72, (128, 576, 576), (331776, 576, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [attn_195], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf91, (128, 48, 576), (27648, 576, 1), 0), out=buf92)
        buf93 = reinterpret_tensor(buf71, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [linear_248], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf92, buf93, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf94 = reinterpret_tensor(buf92, (2654208, 16), (16, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [linear_248], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg63_1, (16, 16), (1, 16), 0), out=buf94)
        del arg63_1
        buf95 = buf69; del buf69  # reuse
        buf96 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [attn_197], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf94, arg64_1, buf95, buf96, 73728, 576, grid=grid(73728), stream=stream0)
        buf97 = reinterpret_tensor(buf94, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [linear_249], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf97, arg64_1, buf95, buf96, 42467328, grid=grid(42467328), stream=stream0)
        del arg64_1
        buf98 = reinterpret_tensor(buf93, (2654208, 16), (16, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [linear_249], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg65_1, (16, 16), (1, 16), 0), out=buf98)
        del arg65_1
        buf99 = reinterpret_tensor(buf97, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [matmul_79], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf98, arg66_1, buf99, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg66_1
        buf100 = reinterpret_tensor(buf91, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [matmul_79], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf89, arg62_1, buf100, 3538944, grid=grid(3538944), stream=stream0)
        del arg62_1
        buf101 = reinterpret_tensor(buf90, (128, 576, 48), (27648, 48, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [matmul_79], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf99, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf100, (128, 576, 48), (27648, 48, 1), 0), out=buf101)
        buf102 = reinterpret_tensor(buf100, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_413], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf101, buf102, 3538944, grid=grid(3538944), stream=stream0)
        buf103 = reinterpret_tensor(buf101, (4608, 768), (768, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf102, (4608, 768), (768, 1), 0), reinterpret_tensor(arg67_1, (768, 768), (1, 768), 0), out=buf103)
        del arg67_1
        buf104 = reinterpret_tensor(buf103, (8, 576, 768), (442368, 768, 1), 0); del buf103  # reuse
        buf108 = reinterpret_tensor(buf102, (8, 576, 768), (442368, 768, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [mul_120, x_412, mul_122, x_416, layer_norm_84], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf104, buf77, arg51_1, buf84, arg57_1, arg58_1, arg68_1, arg70_1, arg71_1, buf108, 4608, 768, grid=grid(4608), stream=stream0)
        del arg51_1
        del arg57_1
        del arg58_1
        del arg68_1
        del arg70_1
        del arg71_1
        buf109 = reinterpret_tensor(buf83, (4608, 3072), (3072, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf108, (4608, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 3072), (1, 768), 0), out=buf109)
        del arg72_1
        buf110 = reinterpret_tensor(buf109, (8, 576, 3072), (1769472, 3072, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_418], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf110, arg73_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg73_1
        buf111 = reinterpret_tensor(buf108, (4608, 768), (768, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg74_1, (3072, 768), (1, 3072), 0), out=buf111)
        del arg74_1
        buf115 = reinterpret_tensor(buf84, (8, 576, 768), (442368, 768, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [mul_123, x_422, layer_norm_85], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf104, arg69_1, buf111, arg75_1, arg77_1, arg78_1, buf115, 4608, 768, grid=grid(4608), stream=stream0)
        del arg77_1
        del arg78_1
        buf116 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf115, (4608, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 2304), (1, 768), 0), out=buf116)
        del arg79_1
        buf117 = reinterpret_tensor(buf115, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [q_42, attn_200], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf116, arg80_1, buf117, 3538944, grid=grid(3538944), stream=stream0)
        buf118 = reinterpret_tensor(buf77, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [attn_200], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf116, arg80_1, buf118, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf119 = reinterpret_tensor(buf99, (128, 576, 576), (331776, 576, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [attn_200], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf117, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf118, (128, 48, 576), (27648, 576, 1), 0), out=buf119)
        buf120 = reinterpret_tensor(buf98, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [linear_254], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf119, buf120, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf121 = reinterpret_tensor(buf119, (2654208, 16), (16, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [linear_254], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg81_1, (16, 16), (1, 16), 0), out=buf121)
        del arg81_1
        buf122 = buf96; del buf96  # reuse
        buf123 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [attn_202], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf121, arg82_1, buf122, buf123, 73728, 576, grid=grid(73728), stream=stream0)
        buf124 = reinterpret_tensor(buf121, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [linear_255], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf124, arg82_1, buf122, buf123, 42467328, grid=grid(42467328), stream=stream0)
        del arg82_1
        buf125 = reinterpret_tensor(buf120, (2654208, 16), (16, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [linear_255], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg83_1, (16, 16), (1, 16), 0), out=buf125)
        del arg83_1
        buf126 = reinterpret_tensor(buf124, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [matmul_81], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf125, arg84_1, buf126, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg84_1
        buf127 = reinterpret_tensor(buf118, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [matmul_81], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf116, arg80_1, buf127, 3538944, grid=grid(3538944), stream=stream0)
        del arg80_1
        buf128 = reinterpret_tensor(buf117, (128, 576, 48), (27648, 48, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [matmul_81], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf126, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf127, (128, 576, 48), (27648, 48, 1), 0), out=buf128)
        buf129 = reinterpret_tensor(buf127, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_423], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf128, buf129, 3538944, grid=grid(3538944), stream=stream0)
        buf130 = reinterpret_tensor(buf128, (4608, 768), (768, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (4608, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 768), (1, 768), 0), out=buf130)
        del arg85_1
        buf131 = reinterpret_tensor(buf130, (8, 576, 768), (442368, 768, 1), 0); del buf130  # reuse
        buf135 = reinterpret_tensor(buf129, (8, 576, 768), (442368, 768, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [mul_123, x_422, mul_125, x_426, layer_norm_86], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf131, buf104, arg69_1, buf111, arg75_1, arg76_1, arg86_1, arg88_1, arg89_1, buf135, 4608, 768, grid=grid(4608), stream=stream0)
        del arg69_1
        del arg75_1
        del arg76_1
        del arg86_1
        del arg88_1
        del arg89_1
        buf136 = reinterpret_tensor(buf110, (4608, 3072), (3072, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (4608, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 3072), (1, 768), 0), out=buf136)
        del arg90_1
        buf137 = reinterpret_tensor(buf136, (8, 576, 3072), (1769472, 3072, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_428], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf137, arg91_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg91_1
        buf138 = reinterpret_tensor(buf135, (4608, 768), (768, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf137, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg92_1, (3072, 768), (1, 3072), 0), out=buf138)
        del arg92_1
        buf142 = reinterpret_tensor(buf111, (8, 576, 768), (442368, 768, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [mul_126, x_432, layer_norm_87], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf131, arg87_1, buf138, arg93_1, arg95_1, arg96_1, buf142, 4608, 768, grid=grid(4608), stream=stream0)
        del arg95_1
        del arg96_1
        buf143 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (4608, 768), (768, 1), 0), reinterpret_tensor(arg97_1, (768, 2304), (1, 768), 0), out=buf143)
        del arg97_1
        buf144 = reinterpret_tensor(buf142, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [q_43, attn_205], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf143, arg98_1, buf144, 3538944, grid=grid(3538944), stream=stream0)
        buf145 = reinterpret_tensor(buf104, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [attn_205], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf143, arg98_1, buf145, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf146 = reinterpret_tensor(buf126, (128, 576, 576), (331776, 576, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [attn_205], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf144, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf145, (128, 48, 576), (27648, 576, 1), 0), out=buf146)
        buf147 = reinterpret_tensor(buf125, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [linear_260], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf146, buf147, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf148 = reinterpret_tensor(buf146, (2654208, 16), (16, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [linear_260], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf147, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg99_1, (16, 16), (1, 16), 0), out=buf148)
        del arg99_1
        buf149 = buf123; del buf123  # reuse
        buf150 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [attn_207], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf148, arg100_1, buf149, buf150, 73728, 576, grid=grid(73728), stream=stream0)
        buf151 = reinterpret_tensor(buf148, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [linear_261], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf151, arg100_1, buf149, buf150, 42467328, grid=grid(42467328), stream=stream0)
        del arg100_1
        buf152 = reinterpret_tensor(buf147, (2654208, 16), (16, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [linear_261], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf151, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg101_1, (16, 16), (1, 16), 0), out=buf152)
        del arg101_1
        buf153 = reinterpret_tensor(buf151, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [matmul_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf152, arg102_1, buf153, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg102_1
        buf154 = reinterpret_tensor(buf145, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [matmul_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf143, arg98_1, buf154, 3538944, grid=grid(3538944), stream=stream0)
        del arg98_1
        buf155 = reinterpret_tensor(buf144, (128, 576, 48), (27648, 48, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [matmul_83], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf153, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf154, (128, 576, 48), (27648, 48, 1), 0), out=buf155)
        buf156 = reinterpret_tensor(buf154, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [x_433], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf155, buf156, 3538944, grid=grid(3538944), stream=stream0)
        buf157 = reinterpret_tensor(buf155, (4608, 768), (768, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (4608, 768), (768, 1), 0), reinterpret_tensor(arg103_1, (768, 768), (1, 768), 0), out=buf157)
        del arg103_1
        buf158 = reinterpret_tensor(buf157, (8, 576, 768), (442368, 768, 1), 0); del buf157  # reuse
        buf162 = reinterpret_tensor(buf156, (8, 576, 768), (442368, 768, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [mul_126, x_432, mul_128, x_436, layer_norm_88], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf158, buf131, arg87_1, buf138, arg93_1, arg94_1, arg104_1, arg106_1, arg107_1, buf162, 4608, 768, grid=grid(4608), stream=stream0)
        del arg104_1
        del arg106_1
        del arg107_1
        del arg87_1
        del arg93_1
        del arg94_1
        buf163 = reinterpret_tensor(buf137, (4608, 3072), (3072, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf162, (4608, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 3072), (1, 768), 0), out=buf163)
        del arg108_1
        buf164 = reinterpret_tensor(buf163, (8, 576, 3072), (1769472, 3072, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_438], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf164, arg109_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg109_1
        buf165 = reinterpret_tensor(buf162, (4608, 768), (768, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg110_1, (3072, 768), (1, 3072), 0), out=buf165)
        del arg110_1
        buf169 = reinterpret_tensor(buf138, (8, 576, 768), (442368, 768, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [mul_129, x_442, layer_norm_89], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf158, arg105_1, buf165, arg111_1, arg113_1, arg114_1, buf169, 4608, 768, grid=grid(4608), stream=stream0)
        del arg113_1
        del arg114_1
        buf170 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (4608, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 2304), (1, 768), 0), out=buf170)
        del arg115_1
        buf171 = reinterpret_tensor(buf169, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [q_44, attn_210], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf170, arg116_1, buf171, 3538944, grid=grid(3538944), stream=stream0)
        buf172 = reinterpret_tensor(buf131, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [attn_210], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf170, arg116_1, buf172, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf173 = reinterpret_tensor(buf153, (128, 576, 576), (331776, 576, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [attn_210], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf171, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf172, (128, 48, 576), (27648, 576, 1), 0), out=buf173)
        buf174 = reinterpret_tensor(buf152, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [linear_266], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf173, buf174, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf175 = reinterpret_tensor(buf173, (2654208, 16), (16, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [linear_266], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf174, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg117_1, (16, 16), (1, 16), 0), out=buf175)
        del arg117_1
        buf176 = buf150; del buf150  # reuse
        buf177 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [attn_212], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf175, arg118_1, buf176, buf177, 73728, 576, grid=grid(73728), stream=stream0)
        buf178 = reinterpret_tensor(buf175, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [linear_267], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf178, arg118_1, buf176, buf177, 42467328, grid=grid(42467328), stream=stream0)
        del arg118_1
        buf179 = reinterpret_tensor(buf174, (2654208, 16), (16, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [linear_267], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg119_1, (16, 16), (1, 16), 0), out=buf179)
        del arg119_1
        buf180 = reinterpret_tensor(buf178, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [matmul_85], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf179, arg120_1, buf180, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg120_1
        buf181 = reinterpret_tensor(buf172, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [matmul_85], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf170, arg116_1, buf181, 3538944, grid=grid(3538944), stream=stream0)
        del arg116_1
        buf182 = reinterpret_tensor(buf171, (128, 576, 48), (27648, 48, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [matmul_85], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf181, (128, 576, 48), (27648, 48, 1), 0), out=buf182)
        buf183 = reinterpret_tensor(buf181, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [x_443], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf182, buf183, 3538944, grid=grid(3538944), stream=stream0)
        buf184 = reinterpret_tensor(buf182, (4608, 768), (768, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf183, (4608, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 768), (1, 768), 0), out=buf184)
        del arg121_1
        buf185 = reinterpret_tensor(buf184, (8, 576, 768), (442368, 768, 1), 0); del buf184  # reuse
        buf189 = reinterpret_tensor(buf183, (8, 576, 768), (442368, 768, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [mul_129, x_442, mul_131, x_446, layer_norm_90], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf185, buf158, arg105_1, buf165, arg111_1, arg112_1, arg122_1, arg124_1, arg125_1, buf189, 4608, 768, grid=grid(4608), stream=stream0)
        del arg105_1
        del arg111_1
        del arg112_1
        del arg122_1
        del arg124_1
        del arg125_1
        buf190 = reinterpret_tensor(buf164, (4608, 3072), (3072, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf189, (4608, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 3072), (1, 768), 0), out=buf190)
        del arg126_1
        buf191 = reinterpret_tensor(buf190, (8, 576, 3072), (1769472, 3072, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_448], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf191, arg127_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg127_1
        buf192 = reinterpret_tensor(buf189, (4608, 768), (768, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg128_1, (3072, 768), (1, 3072), 0), out=buf192)
        del arg128_1
        buf196 = reinterpret_tensor(buf165, (8, 576, 768), (442368, 768, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [mul_132, x_452, layer_norm_91], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf185, arg123_1, buf192, arg129_1, arg131_1, arg132_1, buf196, 4608, 768, grid=grid(4608), stream=stream0)
        del arg131_1
        del arg132_1
        buf197 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (4608, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 2304), (1, 768), 0), out=buf197)
        del arg133_1
        buf198 = reinterpret_tensor(buf196, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [q_45, attn_215], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf197, arg134_1, buf198, 3538944, grid=grid(3538944), stream=stream0)
        buf199 = reinterpret_tensor(buf158, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [attn_215], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf197, arg134_1, buf199, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf200 = reinterpret_tensor(buf180, (128, 576, 576), (331776, 576, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [attn_215], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf199, (128, 48, 576), (27648, 576, 1), 0), out=buf200)
        buf201 = reinterpret_tensor(buf179, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [linear_272], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf200, buf201, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf202 = reinterpret_tensor(buf200, (2654208, 16), (16, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [linear_272], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg135_1, (16, 16), (1, 16), 0), out=buf202)
        del arg135_1
        buf203 = buf177; del buf177  # reuse
        buf204 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [attn_217], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf202, arg136_1, buf203, buf204, 73728, 576, grid=grid(73728), stream=stream0)
        buf205 = reinterpret_tensor(buf202, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [linear_273], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf205, arg136_1, buf203, buf204, 42467328, grid=grid(42467328), stream=stream0)
        del arg136_1
        buf206 = reinterpret_tensor(buf201, (2654208, 16), (16, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [linear_273], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg137_1, (16, 16), (1, 16), 0), out=buf206)
        del arg137_1
        buf207 = reinterpret_tensor(buf205, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [matmul_87], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf206, arg138_1, buf207, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg138_1
        buf208 = reinterpret_tensor(buf199, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [matmul_87], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf197, arg134_1, buf208, 3538944, grid=grid(3538944), stream=stream0)
        del arg134_1
        buf209 = reinterpret_tensor(buf198, (128, 576, 48), (27648, 48, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [matmul_87], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf207, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf208, (128, 576, 48), (27648, 48, 1), 0), out=buf209)
        buf210 = reinterpret_tensor(buf208, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [x_453], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf209, buf210, 3538944, grid=grid(3538944), stream=stream0)
        buf211 = reinterpret_tensor(buf209, (4608, 768), (768, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (4608, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), out=buf211)
        del arg139_1
        buf212 = reinterpret_tensor(buf211, (8, 576, 768), (442368, 768, 1), 0); del buf211  # reuse
        buf216 = reinterpret_tensor(buf210, (8, 576, 768), (442368, 768, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [mul_132, x_452, mul_134, x_456, layer_norm_92], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf212, buf185, arg123_1, buf192, arg129_1, arg130_1, arg140_1, arg142_1, arg143_1, buf216, 4608, 768, grid=grid(4608), stream=stream0)
        del arg123_1
        del arg129_1
        del arg130_1
        del arg140_1
        del arg142_1
        del arg143_1
        buf217 = reinterpret_tensor(buf191, (4608, 3072), (3072, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf216, (4608, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 3072), (1, 768), 0), out=buf217)
        del arg144_1
        buf218 = reinterpret_tensor(buf217, (8, 576, 3072), (1769472, 3072, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_458], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf218, arg145_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg145_1
        buf219 = reinterpret_tensor(buf216, (4608, 768), (768, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg146_1, (3072, 768), (1, 3072), 0), out=buf219)
        del arg146_1
        buf223 = reinterpret_tensor(buf192, (8, 576, 768), (442368, 768, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [mul_135, x_462, layer_norm_93], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf212, arg141_1, buf219, arg147_1, arg149_1, arg150_1, buf223, 4608, 768, grid=grid(4608), stream=stream0)
        del arg149_1
        del arg150_1
        buf224 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (4608, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 2304), (1, 768), 0), out=buf224)
        del arg151_1
        buf225 = reinterpret_tensor(buf223, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [q_46, attn_220], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf224, arg152_1, buf225, 3538944, grid=grid(3538944), stream=stream0)
        buf226 = reinterpret_tensor(buf185, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [attn_220], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf224, arg152_1, buf226, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf227 = reinterpret_tensor(buf207, (128, 576, 576), (331776, 576, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [attn_220], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf225, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf226, (128, 48, 576), (27648, 576, 1), 0), out=buf227)
        buf228 = reinterpret_tensor(buf206, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [linear_278], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf227, buf228, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf229 = reinterpret_tensor(buf227, (2654208, 16), (16, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [linear_278], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg153_1, (16, 16), (1, 16), 0), out=buf229)
        del arg153_1
        buf230 = buf204; del buf204  # reuse
        buf231 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [attn_222], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf229, arg154_1, buf230, buf231, 73728, 576, grid=grid(73728), stream=stream0)
        buf232 = reinterpret_tensor(buf229, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [linear_279], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf232, arg154_1, buf230, buf231, 42467328, grid=grid(42467328), stream=stream0)
        del arg154_1
        buf233 = reinterpret_tensor(buf228, (2654208, 16), (16, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [linear_279], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg155_1, (16, 16), (1, 16), 0), out=buf233)
        del arg155_1
        buf234 = reinterpret_tensor(buf232, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [matmul_89], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf233, arg156_1, buf234, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg156_1
        buf235 = reinterpret_tensor(buf226, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [matmul_89], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf224, arg152_1, buf235, 3538944, grid=grid(3538944), stream=stream0)
        del arg152_1
        buf236 = reinterpret_tensor(buf225, (128, 576, 48), (27648, 48, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [matmul_89], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf234, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf235, (128, 576, 48), (27648, 48, 1), 0), out=buf236)
        buf237 = reinterpret_tensor(buf235, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [x_463], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf236, buf237, 3538944, grid=grid(3538944), stream=stream0)
        buf238 = reinterpret_tensor(buf236, (4608, 768), (768, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf237, (4608, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 768), (1, 768), 0), out=buf238)
        del arg157_1
        buf239 = reinterpret_tensor(buf238, (8, 576, 768), (442368, 768, 1), 0); del buf238  # reuse
        buf243 = reinterpret_tensor(buf237, (8, 576, 768), (442368, 768, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [mul_135, x_462, mul_137, x_466, layer_norm_94], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf239, buf212, arg141_1, buf219, arg147_1, arg148_1, arg158_1, arg160_1, arg161_1, buf243, 4608, 768, grid=grid(4608), stream=stream0)
        del arg141_1
        del arg147_1
        del arg148_1
        del arg158_1
        del arg160_1
        del arg161_1
        buf244 = reinterpret_tensor(buf218, (4608, 3072), (3072, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (4608, 768), (768, 1), 0), reinterpret_tensor(arg162_1, (768, 3072), (1, 768), 0), out=buf244)
        del arg162_1
        buf245 = reinterpret_tensor(buf244, (8, 576, 3072), (1769472, 3072, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [x_468], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf245, arg163_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg163_1
        buf246 = reinterpret_tensor(buf243, (4608, 768), (768, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg164_1, (3072, 768), (1, 3072), 0), out=buf246)
        del arg164_1
        buf250 = reinterpret_tensor(buf219, (8, 576, 768), (442368, 768, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [mul_138, x_472, layer_norm_95], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf239, arg159_1, buf246, arg165_1, arg167_1, arg168_1, buf250, 4608, 768, grid=grid(4608), stream=stream0)
        del arg167_1
        del arg168_1
        buf251 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (4608, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 2304), (1, 768), 0), out=buf251)
        del arg169_1
        buf252 = reinterpret_tensor(buf250, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [q_47, attn_225], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf251, arg170_1, buf252, 3538944, grid=grid(3538944), stream=stream0)
        buf253 = reinterpret_tensor(buf212, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [attn_225], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf251, arg170_1, buf253, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf254 = reinterpret_tensor(buf234, (128, 576, 576), (331776, 576, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [attn_225], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf252, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf253, (128, 48, 576), (27648, 576, 1), 0), out=buf254)
        buf255 = reinterpret_tensor(buf233, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [linear_284], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf254, buf255, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf256 = reinterpret_tensor(buf254, (2654208, 16), (16, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [linear_284], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg171_1, (16, 16), (1, 16), 0), out=buf256)
        del arg171_1
        buf257 = buf231; del buf231  # reuse
        buf258 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [attn_227], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf256, arg172_1, buf257, buf258, 73728, 576, grid=grid(73728), stream=stream0)
        buf259 = reinterpret_tensor(buf256, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [linear_285], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf259, arg172_1, buf257, buf258, 42467328, grid=grid(42467328), stream=stream0)
        del arg172_1
        buf260 = reinterpret_tensor(buf255, (2654208, 16), (16, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [linear_285], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg173_1, (16, 16), (1, 16), 0), out=buf260)
        del arg173_1
        buf261 = reinterpret_tensor(buf259, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [matmul_91], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf260, arg174_1, buf261, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg174_1
        buf262 = reinterpret_tensor(buf253, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [matmul_91], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf251, arg170_1, buf262, 3538944, grid=grid(3538944), stream=stream0)
        del arg170_1
        buf263 = reinterpret_tensor(buf252, (128, 576, 48), (27648, 48, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [matmul_91], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf261, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf262, (128, 576, 48), (27648, 48, 1), 0), out=buf263)
        buf264 = reinterpret_tensor(buf262, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [x_473], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf263, buf264, 3538944, grid=grid(3538944), stream=stream0)
        buf265 = reinterpret_tensor(buf263, (4608, 768), (768, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf264, (4608, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 768), (1, 768), 0), out=buf265)
        del arg175_1
        buf266 = reinterpret_tensor(buf265, (8, 576, 768), (442368, 768, 1), 0); del buf265  # reuse
        buf270 = reinterpret_tensor(buf264, (8, 576, 768), (442368, 768, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [mul_138, x_472, mul_140, x_476, layer_norm_96], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf266, buf239, arg159_1, buf246, arg165_1, arg166_1, arg176_1, arg178_1, arg179_1, buf270, 4608, 768, grid=grid(4608), stream=stream0)
        del arg159_1
        del arg165_1
        del arg166_1
        del arg176_1
        del arg178_1
        del arg179_1
        buf271 = reinterpret_tensor(buf245, (4608, 3072), (3072, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf270, (4608, 768), (768, 1), 0), reinterpret_tensor(arg180_1, (768, 3072), (1, 768), 0), out=buf271)
        del arg180_1
        buf272 = reinterpret_tensor(buf271, (8, 576, 3072), (1769472, 3072, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [x_478], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf272, arg181_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg181_1
        buf273 = reinterpret_tensor(buf270, (4608, 768), (768, 1), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg182_1, (3072, 768), (1, 3072), 0), out=buf273)
        del arg182_1
        buf277 = reinterpret_tensor(buf246, (8, 576, 768), (442368, 768, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [mul_141, x_482, layer_norm_97], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf266, arg177_1, buf273, arg183_1, arg185_1, arg186_1, buf277, 4608, 768, grid=grid(4608), stream=stream0)
        del arg185_1
        del arg186_1
        buf278 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf277, (4608, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 2304), (1, 768), 0), out=buf278)
        del arg187_1
        buf279 = reinterpret_tensor(buf277, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [q_48, attn_230], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf278, arg188_1, buf279, 3538944, grid=grid(3538944), stream=stream0)
        buf280 = reinterpret_tensor(buf239, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [attn_230], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf278, arg188_1, buf280, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf281 = reinterpret_tensor(buf261, (128, 576, 576), (331776, 576, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [attn_230], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf279, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf280, (128, 48, 576), (27648, 576, 1), 0), out=buf281)
        buf282 = reinterpret_tensor(buf260, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [linear_290], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf281, buf282, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf283 = reinterpret_tensor(buf281, (2654208, 16), (16, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [linear_290], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg189_1, (16, 16), (1, 16), 0), out=buf283)
        del arg189_1
        buf284 = buf258; del buf258  # reuse
        buf285 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [attn_232], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf283, arg190_1, buf284, buf285, 73728, 576, grid=grid(73728), stream=stream0)
        buf286 = reinterpret_tensor(buf283, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [linear_291], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf286, arg190_1, buf284, buf285, 42467328, grid=grid(42467328), stream=stream0)
        del arg190_1
        buf287 = reinterpret_tensor(buf282, (2654208, 16), (16, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [linear_291], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg191_1, (16, 16), (1, 16), 0), out=buf287)
        del arg191_1
        buf288 = reinterpret_tensor(buf286, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [matmul_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf287, arg192_1, buf288, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg192_1
        buf289 = reinterpret_tensor(buf280, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [matmul_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf278, arg188_1, buf289, 3538944, grid=grid(3538944), stream=stream0)
        del arg188_1
        buf290 = reinterpret_tensor(buf279, (128, 576, 48), (27648, 48, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [matmul_93], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf288, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf289, (128, 576, 48), (27648, 48, 1), 0), out=buf290)
        buf291 = reinterpret_tensor(buf289, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [x_483], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf290, buf291, 3538944, grid=grid(3538944), stream=stream0)
        buf292 = reinterpret_tensor(buf290, (4608, 768), (768, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf291, (4608, 768), (768, 1), 0), reinterpret_tensor(arg193_1, (768, 768), (1, 768), 0), out=buf292)
        del arg193_1
        buf293 = reinterpret_tensor(buf292, (8, 576, 768), (442368, 768, 1), 0); del buf292  # reuse
        buf297 = reinterpret_tensor(buf291, (8, 576, 768), (442368, 768, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [mul_141, x_482, mul_143, x_486, layer_norm_98], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf293, buf266, arg177_1, buf273, arg183_1, arg184_1, arg194_1, arg196_1, arg197_1, buf297, 4608, 768, grid=grid(4608), stream=stream0)
        del arg177_1
        del arg183_1
        del arg184_1
        del arg194_1
        del arg196_1
        del arg197_1
        buf298 = reinterpret_tensor(buf272, (4608, 3072), (3072, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf297, (4608, 768), (768, 1), 0), reinterpret_tensor(arg198_1, (768, 3072), (1, 768), 0), out=buf298)
        del arg198_1
        buf299 = reinterpret_tensor(buf298, (8, 576, 3072), (1769472, 3072, 1), 0); del buf298  # reuse
        # Topologically Sorted Source Nodes: [x_488], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf299, arg199_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg199_1
        buf300 = reinterpret_tensor(buf297, (4608, 768), (768, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf299, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg200_1, (3072, 768), (1, 3072), 0), out=buf300)
        del arg200_1
        buf304 = reinterpret_tensor(buf273, (8, 576, 768), (442368, 768, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [mul_144, x_492, layer_norm_99], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf293, arg195_1, buf300, arg201_1, arg203_1, arg204_1, buf304, 4608, 768, grid=grid(4608), stream=stream0)
        del arg203_1
        del arg204_1
        buf305 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (4608, 768), (768, 1), 0), reinterpret_tensor(arg205_1, (768, 2304), (1, 768), 0), out=buf305)
        del arg205_1
        buf306 = reinterpret_tensor(buf304, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [q_49, attn_235], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf305, arg206_1, buf306, 3538944, grid=grid(3538944), stream=stream0)
        buf307 = reinterpret_tensor(buf266, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [attn_235], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf305, arg206_1, buf307, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf308 = reinterpret_tensor(buf288, (128, 576, 576), (331776, 576, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [attn_235], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf306, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf307, (128, 48, 576), (27648, 576, 1), 0), out=buf308)
        buf309 = reinterpret_tensor(buf287, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [linear_296], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf308, buf309, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf310 = reinterpret_tensor(buf308, (2654208, 16), (16, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [linear_296], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg207_1, (16, 16), (1, 16), 0), out=buf310)
        del arg207_1
        buf311 = buf285; del buf285  # reuse
        buf312 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [attn_237], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf310, arg208_1, buf311, buf312, 73728, 576, grid=grid(73728), stream=stream0)
        buf313 = reinterpret_tensor(buf310, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [linear_297], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf313, arg208_1, buf311, buf312, 42467328, grid=grid(42467328), stream=stream0)
        del arg208_1
        buf314 = reinterpret_tensor(buf309, (2654208, 16), (16, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [linear_297], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf313, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg209_1, (16, 16), (1, 16), 0), out=buf314)
        del arg209_1
        buf315 = reinterpret_tensor(buf313, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [matmul_95], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf314, arg210_1, buf315, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg210_1
        buf316 = reinterpret_tensor(buf307, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [matmul_95], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf305, arg206_1, buf316, 3538944, grid=grid(3538944), stream=stream0)
        del arg206_1
        buf317 = reinterpret_tensor(buf306, (128, 576, 48), (27648, 48, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [matmul_95], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf315, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf316, (128, 576, 48), (27648, 48, 1), 0), out=buf317)
        buf318 = reinterpret_tensor(buf316, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [x_493], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf317, buf318, 3538944, grid=grid(3538944), stream=stream0)
        buf319 = reinterpret_tensor(buf317, (4608, 768), (768, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf318, (4608, 768), (768, 1), 0), reinterpret_tensor(arg211_1, (768, 768), (1, 768), 0), out=buf319)
        del arg211_1
        buf320 = reinterpret_tensor(buf319, (8, 576, 768), (442368, 768, 1), 0); del buf319  # reuse
        buf324 = reinterpret_tensor(buf318, (8, 576, 768), (442368, 768, 1), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [mul_144, x_492, mul_146, x_496, layer_norm_100], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf320, buf293, arg195_1, buf300, arg201_1, arg202_1, arg212_1, arg214_1, arg215_1, buf324, 4608, 768, grid=grid(4608), stream=stream0)
        del arg195_1
        del arg201_1
        del arg202_1
        del arg212_1
        del arg214_1
        del arg215_1
        buf325 = reinterpret_tensor(buf299, (4608, 3072), (3072, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (4608, 768), (768, 1), 0), reinterpret_tensor(arg216_1, (768, 3072), (1, 768), 0), out=buf325)
        del arg216_1
        buf326 = reinterpret_tensor(buf325, (8, 576, 3072), (1769472, 3072, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [x_498], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf326, arg217_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg217_1
        buf327 = reinterpret_tensor(buf324, (4608, 768), (768, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg218_1, (3072, 768), (1, 3072), 0), out=buf327)
        del arg218_1
        buf331 = reinterpret_tensor(buf300, (8, 576, 768), (442368, 768, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [mul_147, x_502, layer_norm_101], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf320, arg213_1, buf327, arg219_1, arg221_1, arg222_1, buf331, 4608, 768, grid=grid(4608), stream=stream0)
        del arg221_1
        del arg222_1
        buf332 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (4608, 768), (768, 1), 0), reinterpret_tensor(arg223_1, (768, 2304), (1, 768), 0), out=buf332)
        del arg223_1
        buf333 = reinterpret_tensor(buf331, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [q_50, attn_240], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf332, arg224_1, buf333, 3538944, grid=grid(3538944), stream=stream0)
        buf334 = reinterpret_tensor(buf293, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [attn_240], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf332, arg224_1, buf334, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf335 = reinterpret_tensor(buf315, (128, 576, 576), (331776, 576, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [attn_240], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf333, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf334, (128, 48, 576), (27648, 576, 1), 0), out=buf335)
        buf336 = reinterpret_tensor(buf314, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [linear_302], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf335, buf336, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf337 = reinterpret_tensor(buf335, (2654208, 16), (16, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [linear_302], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf336, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg225_1, (16, 16), (1, 16), 0), out=buf337)
        del arg225_1
        buf338 = buf312; del buf312  # reuse
        buf339 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [attn_242], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf337, arg226_1, buf338, buf339, 73728, 576, grid=grid(73728), stream=stream0)
        buf340 = reinterpret_tensor(buf337, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [linear_303], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf340, arg226_1, buf338, buf339, 42467328, grid=grid(42467328), stream=stream0)
        del arg226_1
        buf341 = reinterpret_tensor(buf336, (2654208, 16), (16, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [linear_303], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg227_1, (16, 16), (1, 16), 0), out=buf341)
        del arg227_1
        buf342 = reinterpret_tensor(buf340, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [matmul_97], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf341, arg228_1, buf342, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg228_1
        buf343 = reinterpret_tensor(buf334, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [matmul_97], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf332, arg224_1, buf343, 3538944, grid=grid(3538944), stream=stream0)
        del arg224_1
        buf344 = reinterpret_tensor(buf333, (128, 576, 48), (27648, 48, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [matmul_97], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf342, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf343, (128, 576, 48), (27648, 48, 1), 0), out=buf344)
        buf345 = reinterpret_tensor(buf343, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [x_503], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf344, buf345, 3538944, grid=grid(3538944), stream=stream0)
        buf346 = reinterpret_tensor(buf344, (4608, 768), (768, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf345, (4608, 768), (768, 1), 0), reinterpret_tensor(arg229_1, (768, 768), (1, 768), 0), out=buf346)
        del arg229_1
        buf347 = reinterpret_tensor(buf346, (8, 576, 768), (442368, 768, 1), 0); del buf346  # reuse
        buf351 = reinterpret_tensor(buf345, (8, 576, 768), (442368, 768, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [mul_147, x_502, mul_149, x_506, layer_norm_102], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf347, buf320, arg213_1, buf327, arg219_1, arg220_1, arg230_1, arg232_1, arg233_1, buf351, 4608, 768, grid=grid(4608), stream=stream0)
        del arg213_1
        del arg219_1
        del arg220_1
        del arg230_1
        del arg232_1
        del arg233_1
        buf352 = reinterpret_tensor(buf326, (4608, 3072), (3072, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf351, (4608, 768), (768, 1), 0), reinterpret_tensor(arg234_1, (768, 3072), (1, 768), 0), out=buf352)
        del arg234_1
        buf353 = reinterpret_tensor(buf352, (8, 576, 3072), (1769472, 3072, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [x_508], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf353, arg235_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg235_1
        buf354 = reinterpret_tensor(buf351, (4608, 768), (768, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg236_1, (3072, 768), (1, 3072), 0), out=buf354)
        del arg236_1
        buf358 = reinterpret_tensor(buf327, (8, 576, 768), (442368, 768, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [mul_150, x_512, layer_norm_103], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf347, arg231_1, buf354, arg237_1, arg239_1, arg240_1, buf358, 4608, 768, grid=grid(4608), stream=stream0)
        del arg239_1
        del arg240_1
        buf359 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (4608, 768), (768, 1), 0), reinterpret_tensor(arg241_1, (768, 2304), (1, 768), 0), out=buf359)
        del arg241_1
        buf360 = reinterpret_tensor(buf358, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [q_51, attn_245], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf359, arg242_1, buf360, 3538944, grid=grid(3538944), stream=stream0)
        buf361 = reinterpret_tensor(buf320, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [attn_245], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf359, arg242_1, buf361, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf362 = reinterpret_tensor(buf342, (128, 576, 576), (331776, 576, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [attn_245], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf361, (128, 48, 576), (27648, 576, 1), 0), out=buf362)
        buf363 = reinterpret_tensor(buf341, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [linear_308], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf362, buf363, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf364 = reinterpret_tensor(buf362, (2654208, 16), (16, 1), 0); del buf362  # reuse
        # Topologically Sorted Source Nodes: [linear_308], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg243_1, (16, 16), (1, 16), 0), out=buf364)
        del arg243_1
        buf365 = buf339; del buf339  # reuse
        buf366 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [attn_247], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf364, arg244_1, buf365, buf366, 73728, 576, grid=grid(73728), stream=stream0)
        buf367 = reinterpret_tensor(buf364, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [linear_309], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf367, arg244_1, buf365, buf366, 42467328, grid=grid(42467328), stream=stream0)
        del arg244_1
        buf368 = reinterpret_tensor(buf363, (2654208, 16), (16, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [linear_309], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg245_1, (16, 16), (1, 16), 0), out=buf368)
        del arg245_1
        buf369 = reinterpret_tensor(buf367, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [matmul_99], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf368, arg246_1, buf369, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg246_1
        buf370 = reinterpret_tensor(buf361, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [matmul_99], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf359, arg242_1, buf370, 3538944, grid=grid(3538944), stream=stream0)
        del arg242_1
        buf371 = reinterpret_tensor(buf360, (128, 576, 48), (27648, 48, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [matmul_99], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf370, (128, 576, 48), (27648, 48, 1), 0), out=buf371)
        buf372 = reinterpret_tensor(buf370, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [x_513], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf371, buf372, 3538944, grid=grid(3538944), stream=stream0)
        buf373 = reinterpret_tensor(buf371, (4608, 768), (768, 1), 0); del buf371  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf372, (4608, 768), (768, 1), 0), reinterpret_tensor(arg247_1, (768, 768), (1, 768), 0), out=buf373)
        del arg247_1
        buf374 = reinterpret_tensor(buf373, (8, 576, 768), (442368, 768, 1), 0); del buf373  # reuse
        buf378 = reinterpret_tensor(buf372, (8, 576, 768), (442368, 768, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [mul_150, x_512, mul_152, x_516, layer_norm_104], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf374, buf347, arg231_1, buf354, arg237_1, arg238_1, arg248_1, arg250_1, arg251_1, buf378, 4608, 768, grid=grid(4608), stream=stream0)
        del arg231_1
        del arg237_1
        del arg238_1
        del arg248_1
        del arg250_1
        del arg251_1
        buf379 = reinterpret_tensor(buf353, (4608, 3072), (3072, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf378, (4608, 768), (768, 1), 0), reinterpret_tensor(arg252_1, (768, 3072), (1, 768), 0), out=buf379)
        del arg252_1
        buf380 = reinterpret_tensor(buf379, (8, 576, 3072), (1769472, 3072, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [x_518], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf380, arg253_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg253_1
        buf381 = reinterpret_tensor(buf378, (4608, 768), (768, 1), 0); del buf378  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf380, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg254_1, (3072, 768), (1, 3072), 0), out=buf381)
        del arg254_1
        buf385 = reinterpret_tensor(buf354, (8, 576, 768), (442368, 768, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [mul_153, x_522, layer_norm_105], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf374, arg249_1, buf381, arg255_1, arg257_1, arg258_1, buf385, 4608, 768, grid=grid(4608), stream=stream0)
        del arg257_1
        del arg258_1
        buf386 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf385, (4608, 768), (768, 1), 0), reinterpret_tensor(arg259_1, (768, 2304), (1, 768), 0), out=buf386)
        del arg259_1
        buf387 = reinterpret_tensor(buf385, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [q_52, attn_250], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf386, arg260_1, buf387, 3538944, grid=grid(3538944), stream=stream0)
        buf388 = reinterpret_tensor(buf347, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [attn_250], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf386, arg260_1, buf388, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf389 = reinterpret_tensor(buf369, (128, 576, 576), (331776, 576, 1), 0); del buf369  # reuse
        # Topologically Sorted Source Nodes: [attn_250], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf387, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf388, (128, 48, 576), (27648, 576, 1), 0), out=buf389)
        buf390 = reinterpret_tensor(buf368, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf368  # reuse
        # Topologically Sorted Source Nodes: [linear_314], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf389, buf390, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf391 = reinterpret_tensor(buf389, (2654208, 16), (16, 1), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [linear_314], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg261_1, (16, 16), (1, 16), 0), out=buf391)
        del arg261_1
        buf392 = buf366; del buf366  # reuse
        buf393 = buf365; del buf365  # reuse
        # Topologically Sorted Source Nodes: [attn_252], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf391, arg262_1, buf392, buf393, 73728, 576, grid=grid(73728), stream=stream0)
        buf394 = reinterpret_tensor(buf391, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [linear_315], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf394, arg262_1, buf392, buf393, 42467328, grid=grid(42467328), stream=stream0)
        del arg262_1
        buf395 = reinterpret_tensor(buf390, (2654208, 16), (16, 1), 0); del buf390  # reuse
        # Topologically Sorted Source Nodes: [linear_315], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf394, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg263_1, (16, 16), (1, 16), 0), out=buf395)
        del arg263_1
        buf396 = reinterpret_tensor(buf394, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf394  # reuse
        # Topologically Sorted Source Nodes: [matmul_101], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf395, arg264_1, buf396, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg264_1
        buf397 = reinterpret_tensor(buf388, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [matmul_101], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf386, arg260_1, buf397, 3538944, grid=grid(3538944), stream=stream0)
        del arg260_1
        buf398 = reinterpret_tensor(buf387, (128, 576, 48), (27648, 48, 1), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [matmul_101], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf396, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf397, (128, 576, 48), (27648, 48, 1), 0), out=buf398)
        buf399 = reinterpret_tensor(buf397, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [x_523], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf398, buf399, 3538944, grid=grid(3538944), stream=stream0)
        buf400 = reinterpret_tensor(buf398, (4608, 768), (768, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf399, (4608, 768), (768, 1), 0), reinterpret_tensor(arg265_1, (768, 768), (1, 768), 0), out=buf400)
        del arg265_1
        buf401 = reinterpret_tensor(buf400, (8, 576, 768), (442368, 768, 1), 0); del buf400  # reuse
        buf405 = reinterpret_tensor(buf399, (8, 576, 768), (442368, 768, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [mul_153, x_522, mul_155, x_526, layer_norm_106], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf401, buf374, arg249_1, buf381, arg255_1, arg256_1, arg266_1, arg268_1, arg269_1, buf405, 4608, 768, grid=grid(4608), stream=stream0)
        del arg249_1
        del arg255_1
        del arg256_1
        del arg266_1
        del arg268_1
        del arg269_1
        buf406 = reinterpret_tensor(buf380, (4608, 3072), (3072, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf405, (4608, 768), (768, 1), 0), reinterpret_tensor(arg270_1, (768, 3072), (1, 768), 0), out=buf406)
        del arg270_1
        buf407 = reinterpret_tensor(buf406, (8, 576, 3072), (1769472, 3072, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [x_528], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf407, arg271_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg271_1
        buf408 = reinterpret_tensor(buf405, (4608, 768), (768, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf407, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg272_1, (3072, 768), (1, 3072), 0), out=buf408)
        del arg272_1
        buf412 = reinterpret_tensor(buf381, (8, 576, 768), (442368, 768, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [mul_156, x_532, layer_norm_107], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf401, arg267_1, buf408, arg273_1, arg275_1, arg276_1, buf412, 4608, 768, grid=grid(4608), stream=stream0)
        del arg275_1
        del arg276_1
        buf413 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (4608, 768), (768, 1), 0), reinterpret_tensor(arg277_1, (768, 2304), (1, 768), 0), out=buf413)
        del arg277_1
        buf414 = reinterpret_tensor(buf412, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [q_53, attn_255], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf413, arg278_1, buf414, 3538944, grid=grid(3538944), stream=stream0)
        buf415 = reinterpret_tensor(buf374, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf374  # reuse
        # Topologically Sorted Source Nodes: [attn_255], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf413, arg278_1, buf415, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf416 = reinterpret_tensor(buf396, (128, 576, 576), (331776, 576, 1), 0); del buf396  # reuse
        # Topologically Sorted Source Nodes: [attn_255], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf414, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf415, (128, 48, 576), (27648, 576, 1), 0), out=buf416)
        buf417 = reinterpret_tensor(buf395, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [linear_320], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf416, buf417, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf418 = reinterpret_tensor(buf416, (2654208, 16), (16, 1), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [linear_320], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg279_1, (16, 16), (1, 16), 0), out=buf418)
        del arg279_1
        buf419 = buf393; del buf393  # reuse
        buf420 = buf392; del buf392  # reuse
        # Topologically Sorted Source Nodes: [attn_257], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf418, arg280_1, buf419, buf420, 73728, 576, grid=grid(73728), stream=stream0)
        buf421 = reinterpret_tensor(buf418, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [linear_321], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf421, arg280_1, buf419, buf420, 42467328, grid=grid(42467328), stream=stream0)
        del arg280_1
        buf422 = reinterpret_tensor(buf417, (2654208, 16), (16, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [linear_321], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg281_1, (16, 16), (1, 16), 0), out=buf422)
        del arg281_1
        buf423 = reinterpret_tensor(buf421, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf421  # reuse
        # Topologically Sorted Source Nodes: [matmul_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf422, arg282_1, buf423, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg282_1
        buf424 = reinterpret_tensor(buf415, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [matmul_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf413, arg278_1, buf424, 3538944, grid=grid(3538944), stream=stream0)
        del arg278_1
        buf425 = reinterpret_tensor(buf414, (128, 576, 48), (27648, 48, 1), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [matmul_103], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf423, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf424, (128, 576, 48), (27648, 48, 1), 0), out=buf425)
        buf426 = reinterpret_tensor(buf424, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [x_533], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf425, buf426, 3538944, grid=grid(3538944), stream=stream0)
        buf427 = reinterpret_tensor(buf425, (4608, 768), (768, 1), 0); del buf425  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf426, (4608, 768), (768, 1), 0), reinterpret_tensor(arg283_1, (768, 768), (1, 768), 0), out=buf427)
        del arg283_1
        buf428 = reinterpret_tensor(buf427, (8, 576, 768), (442368, 768, 1), 0); del buf427  # reuse
        buf432 = reinterpret_tensor(buf426, (8, 576, 768), (442368, 768, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [mul_156, x_532, mul_158, x_536, layer_norm_108], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf428, buf401, arg267_1, buf408, arg273_1, arg274_1, arg284_1, arg286_1, arg287_1, buf432, 4608, 768, grid=grid(4608), stream=stream0)
        del arg267_1
        del arg273_1
        del arg274_1
        del arg284_1
        del arg286_1
        del arg287_1
        buf433 = reinterpret_tensor(buf407, (4608, 3072), (3072, 1), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf432, (4608, 768), (768, 1), 0), reinterpret_tensor(arg288_1, (768, 3072), (1, 768), 0), out=buf433)
        del arg288_1
        buf434 = reinterpret_tensor(buf433, (8, 576, 3072), (1769472, 3072, 1), 0); del buf433  # reuse
        # Topologically Sorted Source Nodes: [x_538], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf434, arg289_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg289_1
        buf435 = reinterpret_tensor(buf432, (4608, 768), (768, 1), 0); del buf432  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf434, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg290_1, (3072, 768), (1, 3072), 0), out=buf435)
        del arg290_1
        buf439 = reinterpret_tensor(buf408, (8, 576, 768), (442368, 768, 1), 0); del buf408  # reuse
        # Topologically Sorted Source Nodes: [mul_159, x_542, layer_norm_109], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf428, arg285_1, buf435, arg291_1, arg293_1, arg294_1, buf439, 4608, 768, grid=grid(4608), stream=stream0)
        del arg293_1
        del arg294_1
        buf440 = buf413; del buf413  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf439, (4608, 768), (768, 1), 0), reinterpret_tensor(arg295_1, (768, 2304), (1, 768), 0), out=buf440)
        del arg295_1
        buf441 = reinterpret_tensor(buf439, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [q_54, attn_260], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf440, arg296_1, buf441, 3538944, grid=grid(3538944), stream=stream0)
        buf442 = reinterpret_tensor(buf401, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [attn_260], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf440, arg296_1, buf442, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf443 = reinterpret_tensor(buf423, (128, 576, 576), (331776, 576, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [attn_260], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf441, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf442, (128, 48, 576), (27648, 576, 1), 0), out=buf443)
        buf444 = reinterpret_tensor(buf422, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf422  # reuse
        # Topologically Sorted Source Nodes: [linear_326], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf443, buf444, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf445 = reinterpret_tensor(buf443, (2654208, 16), (16, 1), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [linear_326], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf444, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg297_1, (16, 16), (1, 16), 0), out=buf445)
        del arg297_1
        buf446 = buf420; del buf420  # reuse
        buf447 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [attn_262], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf445, arg298_1, buf446, buf447, 73728, 576, grid=grid(73728), stream=stream0)
        buf448 = reinterpret_tensor(buf445, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf445  # reuse
        # Topologically Sorted Source Nodes: [linear_327], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf448, arg298_1, buf446, buf447, 42467328, grid=grid(42467328), stream=stream0)
        del arg298_1
        buf449 = reinterpret_tensor(buf444, (2654208, 16), (16, 1), 0); del buf444  # reuse
        # Topologically Sorted Source Nodes: [linear_327], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf448, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg299_1, (16, 16), (1, 16), 0), out=buf449)
        del arg299_1
        buf450 = reinterpret_tensor(buf448, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [matmul_105], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf449, arg300_1, buf450, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg300_1
        buf451 = reinterpret_tensor(buf442, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [matmul_105], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf440, arg296_1, buf451, 3538944, grid=grid(3538944), stream=stream0)
        del arg296_1
        buf452 = reinterpret_tensor(buf441, (128, 576, 48), (27648, 48, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [matmul_105], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf450, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf451, (128, 576, 48), (27648, 48, 1), 0), out=buf452)
        buf453 = reinterpret_tensor(buf451, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [x_543], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf452, buf453, 3538944, grid=grid(3538944), stream=stream0)
        buf454 = reinterpret_tensor(buf452, (4608, 768), (768, 1), 0); del buf452  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf453, (4608, 768), (768, 1), 0), reinterpret_tensor(arg301_1, (768, 768), (1, 768), 0), out=buf454)
        del arg301_1
        buf455 = reinterpret_tensor(buf454, (8, 576, 768), (442368, 768, 1), 0); del buf454  # reuse
        buf459 = reinterpret_tensor(buf453, (8, 576, 768), (442368, 768, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [mul_159, x_542, mul_161, x_546, layer_norm_110], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf455, buf428, arg285_1, buf435, arg291_1, arg292_1, arg302_1, arg304_1, arg305_1, buf459, 4608, 768, grid=grid(4608), stream=stream0)
        del arg285_1
        del arg291_1
        del arg292_1
        del arg302_1
        del arg304_1
        del arg305_1
        buf460 = reinterpret_tensor(buf434, (4608, 3072), (3072, 1), 0); del buf434  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf459, (4608, 768), (768, 1), 0), reinterpret_tensor(arg306_1, (768, 3072), (1, 768), 0), out=buf460)
        del arg306_1
        buf461 = reinterpret_tensor(buf460, (8, 576, 3072), (1769472, 3072, 1), 0); del buf460  # reuse
        # Topologically Sorted Source Nodes: [x_548], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf461, arg307_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg307_1
        buf462 = reinterpret_tensor(buf459, (4608, 768), (768, 1), 0); del buf459  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf461, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg308_1, (3072, 768), (1, 3072), 0), out=buf462)
        del arg308_1
        buf466 = reinterpret_tensor(buf435, (8, 576, 768), (442368, 768, 1), 0); del buf435  # reuse
        # Topologically Sorted Source Nodes: [mul_162, x_552, layer_norm_111], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf455, arg303_1, buf462, arg309_1, arg311_1, arg312_1, buf466, 4608, 768, grid=grid(4608), stream=stream0)
        del arg311_1
        del arg312_1
        buf467 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf466, (4608, 768), (768, 1), 0), reinterpret_tensor(arg313_1, (768, 2304), (1, 768), 0), out=buf467)
        del arg313_1
        buf468 = reinterpret_tensor(buf466, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [q_55, attn_265], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf467, arg314_1, buf468, 3538944, grid=grid(3538944), stream=stream0)
        buf469 = reinterpret_tensor(buf428, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf428  # reuse
        # Topologically Sorted Source Nodes: [attn_265], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf467, arg314_1, buf469, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf470 = reinterpret_tensor(buf450, (128, 576, 576), (331776, 576, 1), 0); del buf450  # reuse
        # Topologically Sorted Source Nodes: [attn_265], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf468, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf469, (128, 48, 576), (27648, 576, 1), 0), out=buf470)
        buf471 = reinterpret_tensor(buf449, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [linear_332], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf470, buf471, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf472 = reinterpret_tensor(buf470, (2654208, 16), (16, 1), 0); del buf470  # reuse
        # Topologically Sorted Source Nodes: [linear_332], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf471, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg315_1, (16, 16), (1, 16), 0), out=buf472)
        del arg315_1
        buf473 = buf447; del buf447  # reuse
        buf474 = buf446; del buf446  # reuse
        # Topologically Sorted Source Nodes: [attn_267], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf472, arg316_1, buf473, buf474, 73728, 576, grid=grid(73728), stream=stream0)
        buf475 = reinterpret_tensor(buf472, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf472  # reuse
        # Topologically Sorted Source Nodes: [linear_333], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf475, arg316_1, buf473, buf474, 42467328, grid=grid(42467328), stream=stream0)
        del arg316_1
        buf476 = reinterpret_tensor(buf471, (2654208, 16), (16, 1), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [linear_333], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg317_1, (16, 16), (1, 16), 0), out=buf476)
        del arg317_1
        buf477 = reinterpret_tensor(buf475, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf475  # reuse
        # Topologically Sorted Source Nodes: [matmul_107], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf476, arg318_1, buf477, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg318_1
        buf478 = reinterpret_tensor(buf469, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [matmul_107], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf467, arg314_1, buf478, 3538944, grid=grid(3538944), stream=stream0)
        del arg314_1
        buf479 = reinterpret_tensor(buf468, (128, 576, 48), (27648, 48, 1), 0); del buf468  # reuse
        # Topologically Sorted Source Nodes: [matmul_107], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf477, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf478, (128, 576, 48), (27648, 48, 1), 0), out=buf479)
        buf480 = reinterpret_tensor(buf478, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [x_553], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf479, buf480, 3538944, grid=grid(3538944), stream=stream0)
        buf481 = reinterpret_tensor(buf479, (4608, 768), (768, 1), 0); del buf479  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf480, (4608, 768), (768, 1), 0), reinterpret_tensor(arg319_1, (768, 768), (1, 768), 0), out=buf481)
        del arg319_1
        buf482 = reinterpret_tensor(buf481, (8, 576, 768), (442368, 768, 1), 0); del buf481  # reuse
        buf486 = reinterpret_tensor(buf480, (8, 576, 768), (442368, 768, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [mul_162, x_552, mul_164, x_556, layer_norm_112], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf482, buf455, arg303_1, buf462, arg309_1, arg310_1, arg320_1, arg322_1, arg323_1, buf486, 4608, 768, grid=grid(4608), stream=stream0)
        del arg303_1
        del arg309_1
        del arg310_1
        del arg320_1
        del arg322_1
        del arg323_1
        buf487 = reinterpret_tensor(buf461, (4608, 3072), (3072, 1), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf486, (4608, 768), (768, 1), 0), reinterpret_tensor(arg324_1, (768, 3072), (1, 768), 0), out=buf487)
        del arg324_1
        buf488 = reinterpret_tensor(buf487, (8, 576, 3072), (1769472, 3072, 1), 0); del buf487  # reuse
        # Topologically Sorted Source Nodes: [x_558], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf488, arg325_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg325_1
        buf489 = reinterpret_tensor(buf486, (4608, 768), (768, 1), 0); del buf486  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf488, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg326_1, (3072, 768), (1, 3072), 0), out=buf489)
        del arg326_1
        buf493 = reinterpret_tensor(buf462, (8, 576, 768), (442368, 768, 1), 0); del buf462  # reuse
        # Topologically Sorted Source Nodes: [mul_165, x_562, layer_norm_113], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf482, arg321_1, buf489, arg327_1, arg329_1, arg330_1, buf493, 4608, 768, grid=grid(4608), stream=stream0)
        del arg329_1
        del arg330_1
        buf494 = buf467; del buf467  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf493, (4608, 768), (768, 1), 0), reinterpret_tensor(arg331_1, (768, 2304), (1, 768), 0), out=buf494)
        del arg331_1
        buf495 = reinterpret_tensor(buf493, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf493  # reuse
        # Topologically Sorted Source Nodes: [q_56, attn_270], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf494, arg332_1, buf495, 3538944, grid=grid(3538944), stream=stream0)
        buf496 = reinterpret_tensor(buf455, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf455  # reuse
        # Topologically Sorted Source Nodes: [attn_270], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf494, arg332_1, buf496, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf497 = reinterpret_tensor(buf477, (128, 576, 576), (331776, 576, 1), 0); del buf477  # reuse
        # Topologically Sorted Source Nodes: [attn_270], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf495, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf496, (128, 48, 576), (27648, 576, 1), 0), out=buf497)
        buf498 = reinterpret_tensor(buf476, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [linear_338], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf497, buf498, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf499 = reinterpret_tensor(buf497, (2654208, 16), (16, 1), 0); del buf497  # reuse
        # Topologically Sorted Source Nodes: [linear_338], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf498, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg333_1, (16, 16), (1, 16), 0), out=buf499)
        del arg333_1
        buf500 = buf474; del buf474  # reuse
        buf501 = buf473; del buf473  # reuse
        # Topologically Sorted Source Nodes: [attn_272], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf499, arg334_1, buf500, buf501, 73728, 576, grid=grid(73728), stream=stream0)
        buf502 = reinterpret_tensor(buf499, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf499  # reuse
        # Topologically Sorted Source Nodes: [linear_339], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf502, arg334_1, buf500, buf501, 42467328, grid=grid(42467328), stream=stream0)
        del arg334_1
        buf503 = reinterpret_tensor(buf498, (2654208, 16), (16, 1), 0); del buf498  # reuse
        # Topologically Sorted Source Nodes: [linear_339], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf502, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg335_1, (16, 16), (1, 16), 0), out=buf503)
        del arg335_1
        buf504 = reinterpret_tensor(buf502, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [matmul_109], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf503, arg336_1, buf504, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg336_1
        buf505 = reinterpret_tensor(buf496, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf496  # reuse
        # Topologically Sorted Source Nodes: [matmul_109], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf494, arg332_1, buf505, 3538944, grid=grid(3538944), stream=stream0)
        del arg332_1
        buf506 = reinterpret_tensor(buf495, (128, 576, 48), (27648, 48, 1), 0); del buf495  # reuse
        # Topologically Sorted Source Nodes: [matmul_109], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf504, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf505, (128, 576, 48), (27648, 48, 1), 0), out=buf506)
        buf507 = reinterpret_tensor(buf505, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [x_563], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf506, buf507, 3538944, grid=grid(3538944), stream=stream0)
        buf508 = reinterpret_tensor(buf506, (4608, 768), (768, 1), 0); del buf506  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf507, (4608, 768), (768, 1), 0), reinterpret_tensor(arg337_1, (768, 768), (1, 768), 0), out=buf508)
        del arg337_1
        buf509 = reinterpret_tensor(buf508, (8, 576, 768), (442368, 768, 1), 0); del buf508  # reuse
        buf513 = reinterpret_tensor(buf507, (8, 576, 768), (442368, 768, 1), 0); del buf507  # reuse
        # Topologically Sorted Source Nodes: [mul_165, x_562, mul_167, x_566, layer_norm_114], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf509, buf482, arg321_1, buf489, arg327_1, arg328_1, arg338_1, arg340_1, arg341_1, buf513, 4608, 768, grid=grid(4608), stream=stream0)
        del arg321_1
        del arg327_1
        del arg328_1
        del arg338_1
        del arg340_1
        del arg341_1
        buf514 = reinterpret_tensor(buf488, (4608, 3072), (3072, 1), 0); del buf488  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf513, (4608, 768), (768, 1), 0), reinterpret_tensor(arg342_1, (768, 3072), (1, 768), 0), out=buf514)
        del arg342_1
        buf515 = reinterpret_tensor(buf514, (8, 576, 3072), (1769472, 3072, 1), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [x_568], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf515, arg343_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg343_1
        buf516 = reinterpret_tensor(buf513, (4608, 768), (768, 1), 0); del buf513  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf515, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg344_1, (3072, 768), (1, 3072), 0), out=buf516)
        del arg344_1
        buf520 = reinterpret_tensor(buf489, (8, 576, 768), (442368, 768, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [mul_168, x_572, layer_norm_115], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf509, arg339_1, buf516, arg345_1, arg347_1, arg348_1, buf520, 4608, 768, grid=grid(4608), stream=stream0)
        del arg347_1
        del arg348_1
        buf521 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf520, (4608, 768), (768, 1), 0), reinterpret_tensor(arg349_1, (768, 2304), (1, 768), 0), out=buf521)
        del arg349_1
        buf522 = reinterpret_tensor(buf520, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [q_57, attn_275], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf521, arg350_1, buf522, 3538944, grid=grid(3538944), stream=stream0)
        buf523 = reinterpret_tensor(buf482, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf482  # reuse
        # Topologically Sorted Source Nodes: [attn_275], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf521, arg350_1, buf523, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf524 = reinterpret_tensor(buf504, (128, 576, 576), (331776, 576, 1), 0); del buf504  # reuse
        # Topologically Sorted Source Nodes: [attn_275], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf522, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf523, (128, 48, 576), (27648, 576, 1), 0), out=buf524)
        buf525 = reinterpret_tensor(buf503, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf503  # reuse
        # Topologically Sorted Source Nodes: [linear_344], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf524, buf525, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf526 = reinterpret_tensor(buf524, (2654208, 16), (16, 1), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [linear_344], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg351_1, (16, 16), (1, 16), 0), out=buf526)
        del arg351_1
        buf527 = buf501; del buf501  # reuse
        buf528 = buf500; del buf500  # reuse
        # Topologically Sorted Source Nodes: [attn_277], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf526, arg352_1, buf527, buf528, 73728, 576, grid=grid(73728), stream=stream0)
        buf529 = reinterpret_tensor(buf526, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf526  # reuse
        # Topologically Sorted Source Nodes: [linear_345], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf529, arg352_1, buf527, buf528, 42467328, grid=grid(42467328), stream=stream0)
        del arg352_1
        buf530 = reinterpret_tensor(buf525, (2654208, 16), (16, 1), 0); del buf525  # reuse
        # Topologically Sorted Source Nodes: [linear_345], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf529, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg353_1, (16, 16), (1, 16), 0), out=buf530)
        del arg353_1
        buf531 = reinterpret_tensor(buf529, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf529  # reuse
        # Topologically Sorted Source Nodes: [matmul_111], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf530, arg354_1, buf531, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg354_1
        buf532 = reinterpret_tensor(buf523, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf523  # reuse
        # Topologically Sorted Source Nodes: [matmul_111], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf521, arg350_1, buf532, 3538944, grid=grid(3538944), stream=stream0)
        del arg350_1
        buf533 = reinterpret_tensor(buf522, (128, 576, 48), (27648, 48, 1), 0); del buf522  # reuse
        # Topologically Sorted Source Nodes: [matmul_111], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf531, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf532, (128, 576, 48), (27648, 48, 1), 0), out=buf533)
        buf534 = reinterpret_tensor(buf532, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [x_573], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf533, buf534, 3538944, grid=grid(3538944), stream=stream0)
        buf535 = reinterpret_tensor(buf533, (4608, 768), (768, 1), 0); del buf533  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf534, (4608, 768), (768, 1), 0), reinterpret_tensor(arg355_1, (768, 768), (1, 768), 0), out=buf535)
        del arg355_1
        buf536 = reinterpret_tensor(buf535, (8, 576, 768), (442368, 768, 1), 0); del buf535  # reuse
        buf540 = reinterpret_tensor(buf534, (8, 576, 768), (442368, 768, 1), 0); del buf534  # reuse
        # Topologically Sorted Source Nodes: [mul_168, x_572, mul_170, x_576, layer_norm_116], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf536, buf509, arg339_1, buf516, arg345_1, arg346_1, arg356_1, arg358_1, arg359_1, buf540, 4608, 768, grid=grid(4608), stream=stream0)
        del arg339_1
        del arg345_1
        del arg346_1
        del arg356_1
        del arg358_1
        del arg359_1
        buf541 = reinterpret_tensor(buf515, (4608, 3072), (3072, 1), 0); del buf515  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf540, (4608, 768), (768, 1), 0), reinterpret_tensor(arg360_1, (768, 3072), (1, 768), 0), out=buf541)
        del arg360_1
        buf542 = reinterpret_tensor(buf541, (8, 576, 3072), (1769472, 3072, 1), 0); del buf541  # reuse
        # Topologically Sorted Source Nodes: [x_578], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf542, arg361_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg361_1
        buf543 = reinterpret_tensor(buf540, (4608, 768), (768, 1), 0); del buf540  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf542, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg362_1, (3072, 768), (1, 3072), 0), out=buf543)
        del arg362_1
        buf547 = reinterpret_tensor(buf516, (8, 576, 768), (442368, 768, 1), 0); del buf516  # reuse
        # Topologically Sorted Source Nodes: [mul_171, x_582, layer_norm_117], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf536, arg357_1, buf543, arg363_1, arg365_1, arg366_1, buf547, 4608, 768, grid=grid(4608), stream=stream0)
        del arg365_1
        del arg366_1
        buf548 = buf521; del buf521  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf547, (4608, 768), (768, 1), 0), reinterpret_tensor(arg367_1, (768, 2304), (1, 768), 0), out=buf548)
        del arg367_1
        buf549 = reinterpret_tensor(buf547, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [q_58, attn_280], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf548, arg368_1, buf549, 3538944, grid=grid(3538944), stream=stream0)
        buf550 = reinterpret_tensor(buf509, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf509  # reuse
        # Topologically Sorted Source Nodes: [attn_280], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf548, arg368_1, buf550, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf551 = reinterpret_tensor(buf531, (128, 576, 576), (331776, 576, 1), 0); del buf531  # reuse
        # Topologically Sorted Source Nodes: [attn_280], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf549, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf550, (128, 48, 576), (27648, 576, 1), 0), out=buf551)
        buf552 = reinterpret_tensor(buf530, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [linear_350], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf551, buf552, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf553 = reinterpret_tensor(buf551, (2654208, 16), (16, 1), 0); del buf551  # reuse
        # Topologically Sorted Source Nodes: [linear_350], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf552, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg369_1, (16, 16), (1, 16), 0), out=buf553)
        del arg369_1
        buf554 = buf528; del buf528  # reuse
        buf555 = buf527; del buf527  # reuse
        # Topologically Sorted Source Nodes: [attn_282], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf553, arg370_1, buf554, buf555, 73728, 576, grid=grid(73728), stream=stream0)
        buf556 = reinterpret_tensor(buf553, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf553  # reuse
        # Topologically Sorted Source Nodes: [linear_351], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf556, arg370_1, buf554, buf555, 42467328, grid=grid(42467328), stream=stream0)
        del arg370_1
        buf557 = reinterpret_tensor(buf552, (2654208, 16), (16, 1), 0); del buf552  # reuse
        # Topologically Sorted Source Nodes: [linear_351], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf556, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg371_1, (16, 16), (1, 16), 0), out=buf557)
        del arg371_1
        buf558 = reinterpret_tensor(buf556, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf556  # reuse
        # Topologically Sorted Source Nodes: [matmul_113], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf557, arg372_1, buf558, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg372_1
        buf559 = reinterpret_tensor(buf550, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf550  # reuse
        # Topologically Sorted Source Nodes: [matmul_113], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf548, arg368_1, buf559, 3538944, grid=grid(3538944), stream=stream0)
        del arg368_1
        buf560 = reinterpret_tensor(buf549, (128, 576, 48), (27648, 48, 1), 0); del buf549  # reuse
        # Topologically Sorted Source Nodes: [matmul_113], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf558, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf559, (128, 576, 48), (27648, 48, 1), 0), out=buf560)
        buf561 = reinterpret_tensor(buf559, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf559  # reuse
        # Topologically Sorted Source Nodes: [x_583], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf560, buf561, 3538944, grid=grid(3538944), stream=stream0)
        buf562 = reinterpret_tensor(buf560, (4608, 768), (768, 1), 0); del buf560  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf561, (4608, 768), (768, 1), 0), reinterpret_tensor(arg373_1, (768, 768), (1, 768), 0), out=buf562)
        del arg373_1
        buf563 = reinterpret_tensor(buf562, (8, 576, 768), (442368, 768, 1), 0); del buf562  # reuse
        buf567 = reinterpret_tensor(buf561, (8, 576, 768), (442368, 768, 1), 0); del buf561  # reuse
        # Topologically Sorted Source Nodes: [mul_171, x_582, mul_173, x_586, layer_norm_118], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf563, buf536, arg357_1, buf543, arg363_1, arg364_1, arg374_1, arg376_1, arg377_1, buf567, 4608, 768, grid=grid(4608), stream=stream0)
        del arg357_1
        del arg363_1
        del arg364_1
        del arg374_1
        del arg376_1
        del arg377_1
        buf568 = reinterpret_tensor(buf542, (4608, 3072), (3072, 1), 0); del buf542  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf567, (4608, 768), (768, 1), 0), reinterpret_tensor(arg378_1, (768, 3072), (1, 768), 0), out=buf568)
        del arg378_1
        buf569 = reinterpret_tensor(buf568, (8, 576, 3072), (1769472, 3072, 1), 0); del buf568  # reuse
        # Topologically Sorted Source Nodes: [x_588], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf569, arg379_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg379_1
        buf570 = reinterpret_tensor(buf567, (4608, 768), (768, 1), 0); del buf567  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf569, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg380_1, (3072, 768), (1, 3072), 0), out=buf570)
        del arg380_1
        buf574 = reinterpret_tensor(buf543, (8, 576, 768), (442368, 768, 1), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [mul_174, x_592, layer_norm_119], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf563, arg375_1, buf570, arg381_1, arg383_1, arg384_1, buf574, 4608, 768, grid=grid(4608), stream=stream0)
        del arg383_1
        del arg384_1
        buf575 = buf548; del buf548  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf574, (4608, 768), (768, 1), 0), reinterpret_tensor(arg385_1, (768, 2304), (1, 768), 0), out=buf575)
        del arg385_1
        buf576 = reinterpret_tensor(buf574, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf574  # reuse
        # Topologically Sorted Source Nodes: [q_59, attn_285], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf575, arg386_1, buf576, 3538944, grid=grid(3538944), stream=stream0)
        buf577 = reinterpret_tensor(buf536, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf536  # reuse
        # Topologically Sorted Source Nodes: [attn_285], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf575, arg386_1, buf577, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf578 = reinterpret_tensor(buf558, (128, 576, 576), (331776, 576, 1), 0); del buf558  # reuse
        # Topologically Sorted Source Nodes: [attn_285], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf576, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf577, (128, 48, 576), (27648, 576, 1), 0), out=buf578)
        buf579 = reinterpret_tensor(buf557, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [linear_356], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf578, buf579, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf580 = reinterpret_tensor(buf578, (2654208, 16), (16, 1), 0); del buf578  # reuse
        # Topologically Sorted Source Nodes: [linear_356], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf579, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg387_1, (16, 16), (1, 16), 0), out=buf580)
        del arg387_1
        buf581 = buf555; del buf555  # reuse
        buf582 = buf554; del buf554  # reuse
        # Topologically Sorted Source Nodes: [attn_287], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf580, arg388_1, buf581, buf582, 73728, 576, grid=grid(73728), stream=stream0)
        buf583 = reinterpret_tensor(buf580, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf580  # reuse
        # Topologically Sorted Source Nodes: [linear_357], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf583, arg388_1, buf581, buf582, 42467328, grid=grid(42467328), stream=stream0)
        del arg388_1
        buf584 = reinterpret_tensor(buf579, (2654208, 16), (16, 1), 0); del buf579  # reuse
        # Topologically Sorted Source Nodes: [linear_357], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf583, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg389_1, (16, 16), (1, 16), 0), out=buf584)
        del arg389_1
        buf585 = reinterpret_tensor(buf583, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [matmul_115], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf584, arg390_1, buf585, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg390_1
        buf586 = reinterpret_tensor(buf577, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [matmul_115], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf575, arg386_1, buf586, 3538944, grid=grid(3538944), stream=stream0)
        del arg386_1
        buf587 = reinterpret_tensor(buf576, (128, 576, 48), (27648, 48, 1), 0); del buf576  # reuse
        # Topologically Sorted Source Nodes: [matmul_115], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf585, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf586, (128, 576, 48), (27648, 48, 1), 0), out=buf587)
        buf588 = reinterpret_tensor(buf586, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf586  # reuse
        # Topologically Sorted Source Nodes: [x_593], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf587, buf588, 3538944, grid=grid(3538944), stream=stream0)
        buf589 = reinterpret_tensor(buf587, (4608, 768), (768, 1), 0); del buf587  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf588, (4608, 768), (768, 1), 0), reinterpret_tensor(arg391_1, (768, 768), (1, 768), 0), out=buf589)
        del arg391_1
        buf590 = reinterpret_tensor(buf589, (8, 576, 768), (442368, 768, 1), 0); del buf589  # reuse
        buf594 = reinterpret_tensor(buf588, (8, 576, 768), (442368, 768, 1), 0); del buf588  # reuse
        # Topologically Sorted Source Nodes: [mul_174, x_592, mul_176, x_596, layer_norm_120], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf590, buf563, arg375_1, buf570, arg381_1, arg382_1, arg392_1, arg394_1, arg395_1, buf594, 4608, 768, grid=grid(4608), stream=stream0)
        del arg375_1
        del arg381_1
        del arg382_1
        del arg392_1
        del arg394_1
        del arg395_1
        buf595 = reinterpret_tensor(buf569, (4608, 3072), (3072, 1), 0); del buf569  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf594, (4608, 768), (768, 1), 0), reinterpret_tensor(arg396_1, (768, 3072), (1, 768), 0), out=buf595)
        del arg396_1
        buf596 = reinterpret_tensor(buf595, (8, 576, 3072), (1769472, 3072, 1), 0); del buf595  # reuse
        # Topologically Sorted Source Nodes: [x_598], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf596, arg397_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg397_1
        buf597 = reinterpret_tensor(buf594, (4608, 768), (768, 1), 0); del buf594  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf596, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg398_1, (3072, 768), (1, 3072), 0), out=buf597)
        del arg398_1
        buf601 = reinterpret_tensor(buf570, (8, 576, 768), (442368, 768, 1), 0); del buf570  # reuse
        # Topologically Sorted Source Nodes: [mul_177, x_602, layer_norm_121], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf590, arg393_1, buf597, arg399_1, arg401_1, arg402_1, buf601, 4608, 768, grid=grid(4608), stream=stream0)
        del arg401_1
        del arg402_1
        buf602 = buf575; del buf575  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf601, (4608, 768), (768, 1), 0), reinterpret_tensor(arg403_1, (768, 2304), (1, 768), 0), out=buf602)
        del arg403_1
        buf603 = reinterpret_tensor(buf601, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf601  # reuse
        # Topologically Sorted Source Nodes: [q_60, attn_290], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf602, arg404_1, buf603, 3538944, grid=grid(3538944), stream=stream0)
        buf604 = reinterpret_tensor(buf563, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf563  # reuse
        # Topologically Sorted Source Nodes: [attn_290], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf602, arg404_1, buf604, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf605 = reinterpret_tensor(buf585, (128, 576, 576), (331776, 576, 1), 0); del buf585  # reuse
        # Topologically Sorted Source Nodes: [attn_290], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf603, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf604, (128, 48, 576), (27648, 576, 1), 0), out=buf605)
        buf606 = reinterpret_tensor(buf584, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf584  # reuse
        # Topologically Sorted Source Nodes: [linear_362], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf605, buf606, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf607 = reinterpret_tensor(buf605, (2654208, 16), (16, 1), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [linear_362], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf606, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg405_1, (16, 16), (1, 16), 0), out=buf607)
        del arg405_1
        buf608 = buf582; del buf582  # reuse
        buf609 = buf581; del buf581  # reuse
        # Topologically Sorted Source Nodes: [attn_292], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf607, arg406_1, buf608, buf609, 73728, 576, grid=grid(73728), stream=stream0)
        buf610 = reinterpret_tensor(buf607, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf607  # reuse
        # Topologically Sorted Source Nodes: [linear_363], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf610, arg406_1, buf608, buf609, 42467328, grid=grid(42467328), stream=stream0)
        del arg406_1
        buf611 = reinterpret_tensor(buf606, (2654208, 16), (16, 1), 0); del buf606  # reuse
        # Topologically Sorted Source Nodes: [linear_363], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf610, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg407_1, (16, 16), (1, 16), 0), out=buf611)
        del arg407_1
        buf612 = reinterpret_tensor(buf610, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf610  # reuse
        # Topologically Sorted Source Nodes: [matmul_117], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf611, arg408_1, buf612, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg408_1
        buf613 = reinterpret_tensor(buf604, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf604  # reuse
        # Topologically Sorted Source Nodes: [matmul_117], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf602, arg404_1, buf613, 3538944, grid=grid(3538944), stream=stream0)
        del arg404_1
        buf614 = reinterpret_tensor(buf603, (128, 576, 48), (27648, 48, 1), 0); del buf603  # reuse
        # Topologically Sorted Source Nodes: [matmul_117], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf612, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf613, (128, 576, 48), (27648, 48, 1), 0), out=buf614)
        buf615 = reinterpret_tensor(buf613, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [x_603], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf614, buf615, 3538944, grid=grid(3538944), stream=stream0)
        buf616 = reinterpret_tensor(buf614, (4608, 768), (768, 1), 0); del buf614  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf615, (4608, 768), (768, 1), 0), reinterpret_tensor(arg409_1, (768, 768), (1, 768), 0), out=buf616)
        del arg409_1
        buf617 = reinterpret_tensor(buf616, (8, 576, 768), (442368, 768, 1), 0); del buf616  # reuse
        buf621 = reinterpret_tensor(buf615, (8, 576, 768), (442368, 768, 1), 0); del buf615  # reuse
        # Topologically Sorted Source Nodes: [mul_177, x_602, mul_179, x_606, layer_norm_122], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf617, buf590, arg393_1, buf597, arg399_1, arg400_1, arg410_1, arg412_1, arg413_1, buf621, 4608, 768, grid=grid(4608), stream=stream0)
        del arg393_1
        del arg399_1
        del arg400_1
        del arg410_1
        del arg412_1
        del arg413_1
        buf622 = reinterpret_tensor(buf596, (4608, 3072), (3072, 1), 0); del buf596  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf621, (4608, 768), (768, 1), 0), reinterpret_tensor(arg414_1, (768, 3072), (1, 768), 0), out=buf622)
        del arg414_1
        buf623 = reinterpret_tensor(buf622, (8, 576, 3072), (1769472, 3072, 1), 0); del buf622  # reuse
        # Topologically Sorted Source Nodes: [x_608], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf623, arg415_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg415_1
        buf624 = reinterpret_tensor(buf621, (4608, 768), (768, 1), 0); del buf621  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf623, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg416_1, (3072, 768), (1, 3072), 0), out=buf624)
        del arg416_1
        buf628 = reinterpret_tensor(buf597, (8, 576, 768), (442368, 768, 1), 0); del buf597  # reuse
        # Topologically Sorted Source Nodes: [mul_180, x_612, layer_norm_123], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf617, arg411_1, buf624, arg417_1, arg419_1, arg420_1, buf628, 4608, 768, grid=grid(4608), stream=stream0)
        del arg419_1
        del arg420_1
        buf629 = buf602; del buf602  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf628, (4608, 768), (768, 1), 0), reinterpret_tensor(arg421_1, (768, 2304), (1, 768), 0), out=buf629)
        del arg421_1
        buf630 = reinterpret_tensor(buf628, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf628  # reuse
        # Topologically Sorted Source Nodes: [q_61, attn_295], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf629, arg422_1, buf630, 3538944, grid=grid(3538944), stream=stream0)
        buf631 = reinterpret_tensor(buf590, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf590  # reuse
        # Topologically Sorted Source Nodes: [attn_295], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf629, arg422_1, buf631, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf632 = reinterpret_tensor(buf612, (128, 576, 576), (331776, 576, 1), 0); del buf612  # reuse
        # Topologically Sorted Source Nodes: [attn_295], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf630, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf631, (128, 48, 576), (27648, 576, 1), 0), out=buf632)
        buf633 = reinterpret_tensor(buf611, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf611  # reuse
        # Topologically Sorted Source Nodes: [linear_368], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf632, buf633, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf634 = reinterpret_tensor(buf632, (2654208, 16), (16, 1), 0); del buf632  # reuse
        # Topologically Sorted Source Nodes: [linear_368], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf633, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg423_1, (16, 16), (1, 16), 0), out=buf634)
        del arg423_1
        buf635 = buf609; del buf609  # reuse
        buf636 = buf608; del buf608  # reuse
        # Topologically Sorted Source Nodes: [attn_297], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf634, arg424_1, buf635, buf636, 73728, 576, grid=grid(73728), stream=stream0)
        buf637 = reinterpret_tensor(buf634, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf634  # reuse
        # Topologically Sorted Source Nodes: [linear_369], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf637, arg424_1, buf635, buf636, 42467328, grid=grid(42467328), stream=stream0)
        del arg424_1
        buf638 = reinterpret_tensor(buf633, (2654208, 16), (16, 1), 0); del buf633  # reuse
        # Topologically Sorted Source Nodes: [linear_369], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf637, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg425_1, (16, 16), (1, 16), 0), out=buf638)
        del arg425_1
        buf639 = reinterpret_tensor(buf637, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf637  # reuse
        # Topologically Sorted Source Nodes: [matmul_119], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf638, arg426_1, buf639, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg426_1
        buf640 = reinterpret_tensor(buf631, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf631  # reuse
        # Topologically Sorted Source Nodes: [matmul_119], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf629, arg422_1, buf640, 3538944, grid=grid(3538944), stream=stream0)
        del arg422_1
        buf641 = reinterpret_tensor(buf630, (128, 576, 48), (27648, 48, 1), 0); del buf630  # reuse
        # Topologically Sorted Source Nodes: [matmul_119], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf639, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf640, (128, 576, 48), (27648, 48, 1), 0), out=buf641)
        buf642 = reinterpret_tensor(buf640, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf640  # reuse
        # Topologically Sorted Source Nodes: [x_613], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf641, buf642, 3538944, grid=grid(3538944), stream=stream0)
        buf643 = reinterpret_tensor(buf641, (4608, 768), (768, 1), 0); del buf641  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf642, (4608, 768), (768, 1), 0), reinterpret_tensor(arg427_1, (768, 768), (1, 768), 0), out=buf643)
        del arg427_1
        buf644 = reinterpret_tensor(buf643, (8, 576, 768), (442368, 768, 1), 0); del buf643  # reuse
        buf648 = reinterpret_tensor(buf642, (8, 576, 768), (442368, 768, 1), 0); del buf642  # reuse
        # Topologically Sorted Source Nodes: [mul_180, x_612, mul_182, x_616, layer_norm_124], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf644, buf617, arg411_1, buf624, arg417_1, arg418_1, arg428_1, arg430_1, arg431_1, buf648, 4608, 768, grid=grid(4608), stream=stream0)
        del arg411_1
        del arg417_1
        del arg418_1
        del arg428_1
        del arg430_1
        del arg431_1
        buf649 = reinterpret_tensor(buf623, (4608, 3072), (3072, 1), 0); del buf623  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf648, (4608, 768), (768, 1), 0), reinterpret_tensor(arg432_1, (768, 3072), (1, 768), 0), out=buf649)
        del arg432_1
        buf650 = reinterpret_tensor(buf649, (8, 576, 3072), (1769472, 3072, 1), 0); del buf649  # reuse
        # Topologically Sorted Source Nodes: [x_618], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf650, arg433_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg433_1
        buf651 = reinterpret_tensor(buf648, (4608, 768), (768, 1), 0); del buf648  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf650, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg434_1, (3072, 768), (1, 3072), 0), out=buf651)
        del arg434_1
        buf655 = reinterpret_tensor(buf624, (8, 576, 768), (442368, 768, 1), 0); del buf624  # reuse
        # Topologically Sorted Source Nodes: [mul_183, x_622, layer_norm_125], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf644, arg429_1, buf651, arg435_1, arg437_1, arg438_1, buf655, 4608, 768, grid=grid(4608), stream=stream0)
        del arg437_1
        del arg438_1
        buf656 = buf629; del buf629  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf655, (4608, 768), (768, 1), 0), reinterpret_tensor(arg439_1, (768, 2304), (1, 768), 0), out=buf656)
        del arg439_1
        buf657 = reinterpret_tensor(buf655, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf655  # reuse
        # Topologically Sorted Source Nodes: [q_62, attn_300], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf656, arg440_1, buf657, 3538944, grid=grid(3538944), stream=stream0)
        buf658 = reinterpret_tensor(buf617, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf617  # reuse
        # Topologically Sorted Source Nodes: [attn_300], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf656, arg440_1, buf658, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf659 = reinterpret_tensor(buf639, (128, 576, 576), (331776, 576, 1), 0); del buf639  # reuse
        # Topologically Sorted Source Nodes: [attn_300], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf657, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf658, (128, 48, 576), (27648, 576, 1), 0), out=buf659)
        buf660 = reinterpret_tensor(buf638, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf638  # reuse
        # Topologically Sorted Source Nodes: [linear_374], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf659, buf660, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf661 = reinterpret_tensor(buf659, (2654208, 16), (16, 1), 0); del buf659  # reuse
        # Topologically Sorted Source Nodes: [linear_374], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf660, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg441_1, (16, 16), (1, 16), 0), out=buf661)
        del arg441_1
        buf662 = buf636; del buf636  # reuse
        buf663 = buf635; del buf635  # reuse
        # Topologically Sorted Source Nodes: [attn_302], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf661, arg442_1, buf662, buf663, 73728, 576, grid=grid(73728), stream=stream0)
        buf664 = reinterpret_tensor(buf661, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf661  # reuse
        # Topologically Sorted Source Nodes: [linear_375], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf664, arg442_1, buf662, buf663, 42467328, grid=grid(42467328), stream=stream0)
        del arg442_1
        buf665 = reinterpret_tensor(buf660, (2654208, 16), (16, 1), 0); del buf660  # reuse
        # Topologically Sorted Source Nodes: [linear_375], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf664, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg443_1, (16, 16), (1, 16), 0), out=buf665)
        del arg443_1
        buf666 = reinterpret_tensor(buf664, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf664  # reuse
        # Topologically Sorted Source Nodes: [matmul_121], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf665, arg444_1, buf666, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg444_1
        buf667 = reinterpret_tensor(buf658, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf658  # reuse
        # Topologically Sorted Source Nodes: [matmul_121], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf656, arg440_1, buf667, 3538944, grid=grid(3538944), stream=stream0)
        del arg440_1
        buf668 = reinterpret_tensor(buf657, (128, 576, 48), (27648, 48, 1), 0); del buf657  # reuse
        # Topologically Sorted Source Nodes: [matmul_121], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf666, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf667, (128, 576, 48), (27648, 48, 1), 0), out=buf668)
        buf669 = reinterpret_tensor(buf667, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf667  # reuse
        # Topologically Sorted Source Nodes: [x_623], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf668, buf669, 3538944, grid=grid(3538944), stream=stream0)
        buf670 = reinterpret_tensor(buf668, (4608, 768), (768, 1), 0); del buf668  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf669, (4608, 768), (768, 1), 0), reinterpret_tensor(arg445_1, (768, 768), (1, 768), 0), out=buf670)
        del arg445_1
        buf671 = reinterpret_tensor(buf670, (8, 576, 768), (442368, 768, 1), 0); del buf670  # reuse
        buf675 = reinterpret_tensor(buf669, (8, 576, 768), (442368, 768, 1), 0); del buf669  # reuse
        # Topologically Sorted Source Nodes: [mul_183, x_622, mul_185, x_626, layer_norm_126], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf671, buf644, arg429_1, buf651, arg435_1, arg436_1, arg446_1, arg448_1, arg449_1, buf675, 4608, 768, grid=grid(4608), stream=stream0)
        del arg429_1
        del arg435_1
        del arg436_1
        del arg446_1
        del arg448_1
        del arg449_1
        buf676 = reinterpret_tensor(buf650, (4608, 3072), (3072, 1), 0); del buf650  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf675, (4608, 768), (768, 1), 0), reinterpret_tensor(arg450_1, (768, 3072), (1, 768), 0), out=buf676)
        del arg450_1
        buf677 = reinterpret_tensor(buf676, (8, 576, 3072), (1769472, 3072, 1), 0); del buf676  # reuse
        # Topologically Sorted Source Nodes: [x_628], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf677, arg451_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg451_1
        buf678 = reinterpret_tensor(buf675, (4608, 768), (768, 1), 0); del buf675  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf677, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg452_1, (3072, 768), (1, 3072), 0), out=buf678)
        del arg452_1
        buf682 = reinterpret_tensor(buf651, (8, 576, 768), (442368, 768, 1), 0); del buf651  # reuse
        # Topologically Sorted Source Nodes: [mul_186, x_632, layer_norm_127], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf671, arg447_1, buf678, arg453_1, arg455_1, arg456_1, buf682, 4608, 768, grid=grid(4608), stream=stream0)
        del arg455_1
        del arg456_1
        buf683 = buf656; del buf656  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf682, (4608, 768), (768, 1), 0), reinterpret_tensor(arg457_1, (768, 2304), (1, 768), 0), out=buf683)
        del arg457_1
        buf684 = reinterpret_tensor(buf682, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf682  # reuse
        # Topologically Sorted Source Nodes: [q_63, attn_305], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf683, arg458_1, buf684, 3538944, grid=grid(3538944), stream=stream0)
        buf685 = reinterpret_tensor(buf644, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf644  # reuse
        # Topologically Sorted Source Nodes: [attn_305], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf683, arg458_1, buf685, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf686 = reinterpret_tensor(buf666, (128, 576, 576), (331776, 576, 1), 0); del buf666  # reuse
        # Topologically Sorted Source Nodes: [attn_305], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf684, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf685, (128, 48, 576), (27648, 576, 1), 0), out=buf686)
        buf687 = reinterpret_tensor(buf665, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf665  # reuse
        # Topologically Sorted Source Nodes: [linear_380], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf686, buf687, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf688 = reinterpret_tensor(buf686, (2654208, 16), (16, 1), 0); del buf686  # reuse
        # Topologically Sorted Source Nodes: [linear_380], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf687, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg459_1, (16, 16), (1, 16), 0), out=buf688)
        del arg459_1
        buf689 = buf663; del buf663  # reuse
        buf690 = buf662; del buf662  # reuse
        # Topologically Sorted Source Nodes: [attn_307], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf688, arg460_1, buf689, buf690, 73728, 576, grid=grid(73728), stream=stream0)
        buf691 = reinterpret_tensor(buf688, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf688  # reuse
        # Topologically Sorted Source Nodes: [linear_381], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf691, arg460_1, buf689, buf690, 42467328, grid=grid(42467328), stream=stream0)
        del arg460_1
        buf692 = reinterpret_tensor(buf687, (2654208, 16), (16, 1), 0); del buf687  # reuse
        # Topologically Sorted Source Nodes: [linear_381], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf691, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg461_1, (16, 16), (1, 16), 0), out=buf692)
        del arg461_1
        buf693 = reinterpret_tensor(buf691, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf691  # reuse
        # Topologically Sorted Source Nodes: [matmul_123], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf692, arg462_1, buf693, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg462_1
        buf694 = reinterpret_tensor(buf685, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf685  # reuse
        # Topologically Sorted Source Nodes: [matmul_123], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf683, arg458_1, buf694, 3538944, grid=grid(3538944), stream=stream0)
        del arg458_1
        buf695 = reinterpret_tensor(buf684, (128, 576, 48), (27648, 48, 1), 0); del buf684  # reuse
        # Topologically Sorted Source Nodes: [matmul_123], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf693, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf694, (128, 576, 48), (27648, 48, 1), 0), out=buf695)
        buf696 = reinterpret_tensor(buf694, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf694  # reuse
        # Topologically Sorted Source Nodes: [x_633], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf695, buf696, 3538944, grid=grid(3538944), stream=stream0)
        buf697 = reinterpret_tensor(buf695, (4608, 768), (768, 1), 0); del buf695  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf696, (4608, 768), (768, 1), 0), reinterpret_tensor(arg463_1, (768, 768), (1, 768), 0), out=buf697)
        del arg463_1
        buf698 = reinterpret_tensor(buf697, (8, 576, 768), (442368, 768, 1), 0); del buf697  # reuse
        buf702 = reinterpret_tensor(buf696, (8, 576, 768), (442368, 768, 1), 0); del buf696  # reuse
        # Topologically Sorted Source Nodes: [mul_186, x_632, mul_188, x_636, layer_norm_128], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf698, buf671, arg447_1, buf678, arg453_1, arg454_1, arg464_1, arg466_1, arg467_1, buf702, 4608, 768, grid=grid(4608), stream=stream0)
        del arg447_1
        del arg453_1
        del arg454_1
        del arg464_1
        del arg466_1
        del arg467_1
        buf703 = reinterpret_tensor(buf677, (4608, 3072), (3072, 1), 0); del buf677  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf702, (4608, 768), (768, 1), 0), reinterpret_tensor(arg468_1, (768, 3072), (1, 768), 0), out=buf703)
        del arg468_1
        buf704 = reinterpret_tensor(buf703, (8, 576, 3072), (1769472, 3072, 1), 0); del buf703  # reuse
        # Topologically Sorted Source Nodes: [x_638], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf704, arg469_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg469_1
        buf705 = reinterpret_tensor(buf702, (4608, 768), (768, 1), 0); del buf702  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf704, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg470_1, (3072, 768), (1, 3072), 0), out=buf705)
        del arg470_1
        buf709 = reinterpret_tensor(buf678, (8, 576, 768), (442368, 768, 1), 0); del buf678  # reuse
        # Topologically Sorted Source Nodes: [mul_189, x_642, layer_norm_129], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf698, arg465_1, buf705, arg471_1, arg473_1, arg474_1, buf709, 4608, 768, grid=grid(4608), stream=stream0)
        del arg473_1
        del arg474_1
        buf710 = buf683; del buf683  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf709, (4608, 768), (768, 1), 0), reinterpret_tensor(arg475_1, (768, 2304), (1, 768), 0), out=buf710)
        del arg475_1
        buf711 = reinterpret_tensor(buf709, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf709  # reuse
        # Topologically Sorted Source Nodes: [q_64, attn_310], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf710, arg476_1, buf711, 3538944, grid=grid(3538944), stream=stream0)
        buf712 = reinterpret_tensor(buf671, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf671  # reuse
        # Topologically Sorted Source Nodes: [attn_310], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf710, arg476_1, buf712, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf713 = reinterpret_tensor(buf693, (128, 576, 576), (331776, 576, 1), 0); del buf693  # reuse
        # Topologically Sorted Source Nodes: [attn_310], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf711, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf712, (128, 48, 576), (27648, 576, 1), 0), out=buf713)
        buf714 = reinterpret_tensor(buf692, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf692  # reuse
        # Topologically Sorted Source Nodes: [linear_386], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf713, buf714, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf715 = reinterpret_tensor(buf713, (2654208, 16), (16, 1), 0); del buf713  # reuse
        # Topologically Sorted Source Nodes: [linear_386], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf714, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg477_1, (16, 16), (1, 16), 0), out=buf715)
        del arg477_1
        buf716 = buf690; del buf690  # reuse
        buf717 = buf689; del buf689  # reuse
        # Topologically Sorted Source Nodes: [attn_312], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf715, arg478_1, buf716, buf717, 73728, 576, grid=grid(73728), stream=stream0)
        buf718 = reinterpret_tensor(buf715, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf715  # reuse
        # Topologically Sorted Source Nodes: [linear_387], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf718, arg478_1, buf716, buf717, 42467328, grid=grid(42467328), stream=stream0)
        del arg478_1
        buf719 = reinterpret_tensor(buf714, (2654208, 16), (16, 1), 0); del buf714  # reuse
        # Topologically Sorted Source Nodes: [linear_387], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf718, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg479_1, (16, 16), (1, 16), 0), out=buf719)
        del arg479_1
        buf720 = reinterpret_tensor(buf718, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf718  # reuse
        # Topologically Sorted Source Nodes: [matmul_125], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf719, arg480_1, buf720, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg480_1
        buf721 = reinterpret_tensor(buf712, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf712  # reuse
        # Topologically Sorted Source Nodes: [matmul_125], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf710, arg476_1, buf721, 3538944, grid=grid(3538944), stream=stream0)
        del arg476_1
        buf722 = reinterpret_tensor(buf711, (128, 576, 48), (27648, 48, 1), 0); del buf711  # reuse
        # Topologically Sorted Source Nodes: [matmul_125], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf720, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf721, (128, 576, 48), (27648, 48, 1), 0), out=buf722)
        buf723 = reinterpret_tensor(buf721, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf721  # reuse
        # Topologically Sorted Source Nodes: [x_643], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf722, buf723, 3538944, grid=grid(3538944), stream=stream0)
        buf724 = reinterpret_tensor(buf722, (4608, 768), (768, 1), 0); del buf722  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf723, (4608, 768), (768, 1), 0), reinterpret_tensor(arg481_1, (768, 768), (1, 768), 0), out=buf724)
        del arg481_1
        buf725 = reinterpret_tensor(buf724, (8, 576, 768), (442368, 768, 1), 0); del buf724  # reuse
        buf729 = reinterpret_tensor(buf723, (8, 576, 768), (442368, 768, 1), 0); del buf723  # reuse
        # Topologically Sorted Source Nodes: [mul_189, x_642, mul_191, x_646, layer_norm_130], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf725, buf698, arg465_1, buf705, arg471_1, arg472_1, arg482_1, arg484_1, arg485_1, buf729, 4608, 768, grid=grid(4608), stream=stream0)
        del arg465_1
        del arg471_1
        del arg472_1
        del arg482_1
        del arg484_1
        del arg485_1
        buf730 = reinterpret_tensor(buf704, (4608, 3072), (3072, 1), 0); del buf704  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf729, (4608, 768), (768, 1), 0), reinterpret_tensor(arg486_1, (768, 3072), (1, 768), 0), out=buf730)
        del arg486_1
        buf731 = reinterpret_tensor(buf730, (8, 576, 3072), (1769472, 3072, 1), 0); del buf730  # reuse
        # Topologically Sorted Source Nodes: [x_648], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf731, arg487_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg487_1
        buf732 = reinterpret_tensor(buf729, (4608, 768), (768, 1), 0); del buf729  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf731, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg488_1, (3072, 768), (1, 3072), 0), out=buf732)
        del arg488_1
        buf736 = reinterpret_tensor(buf705, (8, 576, 768), (442368, 768, 1), 0); del buf705  # reuse
        # Topologically Sorted Source Nodes: [mul_192, x_652, layer_norm_131], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf725, arg483_1, buf732, arg489_1, arg491_1, arg492_1, buf736, 4608, 768, grid=grid(4608), stream=stream0)
        del arg491_1
        del arg492_1
        buf737 = buf710; del buf710  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf736, (4608, 768), (768, 1), 0), reinterpret_tensor(arg493_1, (768, 2304), (1, 768), 0), out=buf737)
        del arg493_1
        buf738 = reinterpret_tensor(buf736, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf736  # reuse
        # Topologically Sorted Source Nodes: [q_65, attn_315], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf737, arg494_1, buf738, 3538944, grid=grid(3538944), stream=stream0)
        buf739 = reinterpret_tensor(buf698, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf698  # reuse
        # Topologically Sorted Source Nodes: [attn_315], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf737, arg494_1, buf739, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf740 = reinterpret_tensor(buf720, (128, 576, 576), (331776, 576, 1), 0); del buf720  # reuse
        # Topologically Sorted Source Nodes: [attn_315], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf738, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf739, (128, 48, 576), (27648, 576, 1), 0), out=buf740)
        buf741 = reinterpret_tensor(buf719, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf719  # reuse
        # Topologically Sorted Source Nodes: [linear_392], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf740, buf741, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf742 = reinterpret_tensor(buf740, (2654208, 16), (16, 1), 0); del buf740  # reuse
        # Topologically Sorted Source Nodes: [linear_392], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg495_1, (16, 16), (1, 16), 0), out=buf742)
        del arg495_1
        buf743 = buf717; del buf717  # reuse
        buf744 = buf716; del buf716  # reuse
        # Topologically Sorted Source Nodes: [attn_317], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf742, arg496_1, buf743, buf744, 73728, 576, grid=grid(73728), stream=stream0)
        buf745 = reinterpret_tensor(buf742, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf742  # reuse
        # Topologically Sorted Source Nodes: [linear_393], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf745, arg496_1, buf743, buf744, 42467328, grid=grid(42467328), stream=stream0)
        del arg496_1
        buf746 = reinterpret_tensor(buf741, (2654208, 16), (16, 1), 0); del buf741  # reuse
        # Topologically Sorted Source Nodes: [linear_393], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf745, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg497_1, (16, 16), (1, 16), 0), out=buf746)
        del arg497_1
        buf747 = reinterpret_tensor(buf745, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf745  # reuse
        # Topologically Sorted Source Nodes: [matmul_127], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf746, arg498_1, buf747, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg498_1
        buf748 = reinterpret_tensor(buf739, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf739  # reuse
        # Topologically Sorted Source Nodes: [matmul_127], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf737, arg494_1, buf748, 3538944, grid=grid(3538944), stream=stream0)
        del arg494_1
        buf749 = reinterpret_tensor(buf738, (128, 576, 48), (27648, 48, 1), 0); del buf738  # reuse
        # Topologically Sorted Source Nodes: [matmul_127], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf747, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf748, (128, 576, 48), (27648, 48, 1), 0), out=buf749)
        buf750 = reinterpret_tensor(buf748, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf748  # reuse
        # Topologically Sorted Source Nodes: [x_653], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf749, buf750, 3538944, grid=grid(3538944), stream=stream0)
        buf751 = reinterpret_tensor(buf749, (4608, 768), (768, 1), 0); del buf749  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf750, (4608, 768), (768, 1), 0), reinterpret_tensor(arg499_1, (768, 768), (1, 768), 0), out=buf751)
        del arg499_1
        buf752 = reinterpret_tensor(buf751, (8, 576, 768), (442368, 768, 1), 0); del buf751  # reuse
        buf756 = reinterpret_tensor(buf750, (8, 576, 768), (442368, 768, 1), 0); del buf750  # reuse
        # Topologically Sorted Source Nodes: [mul_192, x_652, mul_194, x_656, layer_norm_132], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf752, buf725, arg483_1, buf732, arg489_1, arg490_1, arg500_1, arg502_1, arg503_1, buf756, 4608, 768, grid=grid(4608), stream=stream0)
        del arg483_1
        del arg489_1
        del arg490_1
        del arg500_1
        del arg502_1
        del arg503_1
        buf757 = reinterpret_tensor(buf731, (4608, 3072), (3072, 1), 0); del buf731  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf756, (4608, 768), (768, 1), 0), reinterpret_tensor(arg504_1, (768, 3072), (1, 768), 0), out=buf757)
        del arg504_1
        buf758 = reinterpret_tensor(buf757, (8, 576, 3072), (1769472, 3072, 1), 0); del buf757  # reuse
        # Topologically Sorted Source Nodes: [x_658], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf758, arg505_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg505_1
        buf759 = reinterpret_tensor(buf756, (4608, 768), (768, 1), 0); del buf756  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf758, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg506_1, (3072, 768), (1, 3072), 0), out=buf759)
        del arg506_1
        buf763 = reinterpret_tensor(buf732, (8, 576, 768), (442368, 768, 1), 0); del buf732  # reuse
        # Topologically Sorted Source Nodes: [mul_195, x_662, layer_norm_133], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf752, arg501_1, buf759, arg507_1, arg509_1, arg510_1, buf763, 4608, 768, grid=grid(4608), stream=stream0)
        del arg509_1
        del arg510_1
        buf764 = buf737; del buf737  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf763, (4608, 768), (768, 1), 0), reinterpret_tensor(arg511_1, (768, 2304), (1, 768), 0), out=buf764)
        del arg511_1
        buf765 = reinterpret_tensor(buf763, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf763  # reuse
        # Topologically Sorted Source Nodes: [q_66, attn_320], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf764, arg512_1, buf765, 3538944, grid=grid(3538944), stream=stream0)
        buf766 = reinterpret_tensor(buf725, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf725  # reuse
        # Topologically Sorted Source Nodes: [attn_320], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf764, arg512_1, buf766, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf767 = reinterpret_tensor(buf747, (128, 576, 576), (331776, 576, 1), 0); del buf747  # reuse
        # Topologically Sorted Source Nodes: [attn_320], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf765, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf766, (128, 48, 576), (27648, 576, 1), 0), out=buf767)
        buf768 = reinterpret_tensor(buf746, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf746  # reuse
        # Topologically Sorted Source Nodes: [linear_398], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf767, buf768, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf769 = reinterpret_tensor(buf767, (2654208, 16), (16, 1), 0); del buf767  # reuse
        # Topologically Sorted Source Nodes: [linear_398], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf768, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg513_1, (16, 16), (1, 16), 0), out=buf769)
        del arg513_1
        buf770 = buf744; del buf744  # reuse
        buf771 = buf743; del buf743  # reuse
        # Topologically Sorted Source Nodes: [attn_322], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf769, arg514_1, buf770, buf771, 73728, 576, grid=grid(73728), stream=stream0)
        buf772 = reinterpret_tensor(buf769, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf769  # reuse
        # Topologically Sorted Source Nodes: [linear_399], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf772, arg514_1, buf770, buf771, 42467328, grid=grid(42467328), stream=stream0)
        del arg514_1
        buf773 = reinterpret_tensor(buf768, (2654208, 16), (16, 1), 0); del buf768  # reuse
        # Topologically Sorted Source Nodes: [linear_399], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg515_1, (16, 16), (1, 16), 0), out=buf773)
        del arg515_1
        buf774 = reinterpret_tensor(buf772, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf772  # reuse
        # Topologically Sorted Source Nodes: [matmul_129], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf773, arg516_1, buf774, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg516_1
        buf775 = reinterpret_tensor(buf766, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf766  # reuse
        # Topologically Sorted Source Nodes: [matmul_129], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf764, arg512_1, buf775, 3538944, grid=grid(3538944), stream=stream0)
        del arg512_1
        buf776 = reinterpret_tensor(buf765, (128, 576, 48), (27648, 48, 1), 0); del buf765  # reuse
        # Topologically Sorted Source Nodes: [matmul_129], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf774, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf775, (128, 576, 48), (27648, 48, 1), 0), out=buf776)
        buf777 = reinterpret_tensor(buf775, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf775  # reuse
        # Topologically Sorted Source Nodes: [x_663], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf776, buf777, 3538944, grid=grid(3538944), stream=stream0)
        buf778 = reinterpret_tensor(buf776, (4608, 768), (768, 1), 0); del buf776  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf777, (4608, 768), (768, 1), 0), reinterpret_tensor(arg517_1, (768, 768), (1, 768), 0), out=buf778)
        del arg517_1
        buf779 = reinterpret_tensor(buf778, (8, 576, 768), (442368, 768, 1), 0); del buf778  # reuse
        buf783 = reinterpret_tensor(buf777, (8, 576, 768), (442368, 768, 1), 0); del buf777  # reuse
        # Topologically Sorted Source Nodes: [mul_195, x_662, mul_197, x_666, layer_norm_134], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf779, buf752, arg501_1, buf759, arg507_1, arg508_1, arg518_1, arg520_1, arg521_1, buf783, 4608, 768, grid=grid(4608), stream=stream0)
        del arg501_1
        del arg507_1
        del arg508_1
        del arg518_1
        del arg520_1
        del arg521_1
        buf784 = reinterpret_tensor(buf758, (4608, 3072), (3072, 1), 0); del buf758  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf783, (4608, 768), (768, 1), 0), reinterpret_tensor(arg522_1, (768, 3072), (1, 768), 0), out=buf784)
        del arg522_1
        buf785 = reinterpret_tensor(buf784, (8, 576, 3072), (1769472, 3072, 1), 0); del buf784  # reuse
        # Topologically Sorted Source Nodes: [x_668], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf785, arg523_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg523_1
        buf786 = reinterpret_tensor(buf783, (4608, 768), (768, 1), 0); del buf783  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf785, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg524_1, (3072, 768), (1, 3072), 0), out=buf786)
        del arg524_1
        buf790 = reinterpret_tensor(buf759, (8, 576, 768), (442368, 768, 1), 0); del buf759  # reuse
        # Topologically Sorted Source Nodes: [mul_198, x_672, layer_norm_135], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf779, arg519_1, buf786, arg525_1, arg527_1, arg528_1, buf790, 4608, 768, grid=grid(4608), stream=stream0)
        del arg527_1
        del arg528_1
        buf791 = buf764; del buf764  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf790, (4608, 768), (768, 1), 0), reinterpret_tensor(arg529_1, (768, 2304), (1, 768), 0), out=buf791)
        del arg529_1
        buf792 = reinterpret_tensor(buf790, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf790  # reuse
        # Topologically Sorted Source Nodes: [q_67, attn_325], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf791, arg530_1, buf792, 3538944, grid=grid(3538944), stream=stream0)
        buf793 = reinterpret_tensor(buf752, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf752  # reuse
        # Topologically Sorted Source Nodes: [attn_325], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf791, arg530_1, buf793, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf794 = reinterpret_tensor(buf774, (128, 576, 576), (331776, 576, 1), 0); del buf774  # reuse
        # Topologically Sorted Source Nodes: [attn_325], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf792, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf793, (128, 48, 576), (27648, 576, 1), 0), out=buf794)
        buf795 = reinterpret_tensor(buf773, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf773  # reuse
        # Topologically Sorted Source Nodes: [linear_404], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf794, buf795, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf796 = reinterpret_tensor(buf794, (2654208, 16), (16, 1), 0); del buf794  # reuse
        # Topologically Sorted Source Nodes: [linear_404], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf795, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg531_1, (16, 16), (1, 16), 0), out=buf796)
        del arg531_1
        buf797 = buf771; del buf771  # reuse
        buf798 = buf770; del buf770  # reuse
        # Topologically Sorted Source Nodes: [attn_327], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf796, arg532_1, buf797, buf798, 73728, 576, grid=grid(73728), stream=stream0)
        buf799 = reinterpret_tensor(buf796, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf796  # reuse
        # Topologically Sorted Source Nodes: [linear_405], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf799, arg532_1, buf797, buf798, 42467328, grid=grid(42467328), stream=stream0)
        del arg532_1
        buf800 = reinterpret_tensor(buf795, (2654208, 16), (16, 1), 0); del buf795  # reuse
        # Topologically Sorted Source Nodes: [linear_405], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf799, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg533_1, (16, 16), (1, 16), 0), out=buf800)
        del arg533_1
        buf801 = reinterpret_tensor(buf799, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf799  # reuse
        # Topologically Sorted Source Nodes: [matmul_131], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf800, arg534_1, buf801, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg534_1
        buf802 = reinterpret_tensor(buf793, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf793  # reuse
        # Topologically Sorted Source Nodes: [matmul_131], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf791, arg530_1, buf802, 3538944, grid=grid(3538944), stream=stream0)
        del arg530_1
        buf803 = reinterpret_tensor(buf792, (128, 576, 48), (27648, 48, 1), 0); del buf792  # reuse
        # Topologically Sorted Source Nodes: [matmul_131], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf801, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf802, (128, 576, 48), (27648, 48, 1), 0), out=buf803)
        buf804 = reinterpret_tensor(buf802, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf802  # reuse
        # Topologically Sorted Source Nodes: [x_673], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf803, buf804, 3538944, grid=grid(3538944), stream=stream0)
        buf805 = reinterpret_tensor(buf803, (4608, 768), (768, 1), 0); del buf803  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf804, (4608, 768), (768, 1), 0), reinterpret_tensor(arg535_1, (768, 768), (1, 768), 0), out=buf805)
        del arg535_1
        buf806 = reinterpret_tensor(buf805, (8, 576, 768), (442368, 768, 1), 0); del buf805  # reuse
        buf810 = reinterpret_tensor(buf804, (8, 576, 768), (442368, 768, 1), 0); del buf804  # reuse
        # Topologically Sorted Source Nodes: [mul_198, x_672, mul_200, x_676, layer_norm_136], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf806, buf779, arg519_1, buf786, arg525_1, arg526_1, arg536_1, arg538_1, arg539_1, buf810, 4608, 768, grid=grid(4608), stream=stream0)
        del arg519_1
        del arg525_1
        del arg526_1
        del arg536_1
        del arg538_1
        del arg539_1
        buf811 = reinterpret_tensor(buf785, (4608, 3072), (3072, 1), 0); del buf785  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf810, (4608, 768), (768, 1), 0), reinterpret_tensor(arg540_1, (768, 3072), (1, 768), 0), out=buf811)
        del arg540_1
        buf812 = reinterpret_tensor(buf811, (8, 576, 3072), (1769472, 3072, 1), 0); del buf811  # reuse
        # Topologically Sorted Source Nodes: [x_678], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf812, arg541_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg541_1
        buf813 = reinterpret_tensor(buf810, (4608, 768), (768, 1), 0); del buf810  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf812, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg542_1, (3072, 768), (1, 3072), 0), out=buf813)
        del arg542_1
        buf817 = reinterpret_tensor(buf786, (8, 576, 768), (442368, 768, 1), 0); del buf786  # reuse
        # Topologically Sorted Source Nodes: [mul_201, x_682, layer_norm_137], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf806, arg537_1, buf813, arg543_1, arg545_1, arg546_1, buf817, 4608, 768, grid=grid(4608), stream=stream0)
        del arg545_1
        del arg546_1
        buf818 = buf791; del buf791  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf817, (4608, 768), (768, 1), 0), reinterpret_tensor(arg547_1, (768, 2304), (1, 768), 0), out=buf818)
        del arg547_1
        buf819 = reinterpret_tensor(buf817, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf817  # reuse
        # Topologically Sorted Source Nodes: [q_68, attn_330], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf818, arg548_1, buf819, 3538944, grid=grid(3538944), stream=stream0)
        buf820 = reinterpret_tensor(buf779, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf779  # reuse
        # Topologically Sorted Source Nodes: [attn_330], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf818, arg548_1, buf820, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf821 = reinterpret_tensor(buf801, (128, 576, 576), (331776, 576, 1), 0); del buf801  # reuse
        # Topologically Sorted Source Nodes: [attn_330], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf819, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf820, (128, 48, 576), (27648, 576, 1), 0), out=buf821)
        buf822 = reinterpret_tensor(buf800, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf800  # reuse
        # Topologically Sorted Source Nodes: [linear_410], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf821, buf822, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf823 = reinterpret_tensor(buf821, (2654208, 16), (16, 1), 0); del buf821  # reuse
        # Topologically Sorted Source Nodes: [linear_410], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf822, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg549_1, (16, 16), (1, 16), 0), out=buf823)
        del arg549_1
        buf824 = buf798; del buf798  # reuse
        buf825 = buf797; del buf797  # reuse
        # Topologically Sorted Source Nodes: [attn_332], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf823, arg550_1, buf824, buf825, 73728, 576, grid=grid(73728), stream=stream0)
        buf826 = reinterpret_tensor(buf823, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf823  # reuse
        # Topologically Sorted Source Nodes: [linear_411], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf826, arg550_1, buf824, buf825, 42467328, grid=grid(42467328), stream=stream0)
        del arg550_1
        buf827 = reinterpret_tensor(buf822, (2654208, 16), (16, 1), 0); del buf822  # reuse
        # Topologically Sorted Source Nodes: [linear_411], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf826, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg551_1, (16, 16), (1, 16), 0), out=buf827)
        del arg551_1
        buf828 = reinterpret_tensor(buf826, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf826  # reuse
        # Topologically Sorted Source Nodes: [matmul_133], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf827, arg552_1, buf828, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg552_1
        buf829 = reinterpret_tensor(buf820, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf820  # reuse
        # Topologically Sorted Source Nodes: [matmul_133], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf818, arg548_1, buf829, 3538944, grid=grid(3538944), stream=stream0)
        del arg548_1
        buf830 = reinterpret_tensor(buf819, (128, 576, 48), (27648, 48, 1), 0); del buf819  # reuse
        # Topologically Sorted Source Nodes: [matmul_133], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf828, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf829, (128, 576, 48), (27648, 48, 1), 0), out=buf830)
        buf831 = reinterpret_tensor(buf829, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf829  # reuse
        # Topologically Sorted Source Nodes: [x_683], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf830, buf831, 3538944, grid=grid(3538944), stream=stream0)
        buf832 = reinterpret_tensor(buf830, (4608, 768), (768, 1), 0); del buf830  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf831, (4608, 768), (768, 1), 0), reinterpret_tensor(arg553_1, (768, 768), (1, 768), 0), out=buf832)
        del arg553_1
        buf833 = reinterpret_tensor(buf832, (8, 576, 768), (442368, 768, 1), 0); del buf832  # reuse
        buf837 = reinterpret_tensor(buf831, (8, 576, 768), (442368, 768, 1), 0); del buf831  # reuse
        # Topologically Sorted Source Nodes: [mul_201, x_682, mul_203, x_686, layer_norm_138], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf833, buf806, arg537_1, buf813, arg543_1, arg544_1, arg554_1, arg556_1, arg557_1, buf837, 4608, 768, grid=grid(4608), stream=stream0)
        del arg537_1
        del arg543_1
        del arg544_1
        del arg554_1
        del arg556_1
        del arg557_1
        buf838 = reinterpret_tensor(buf812, (4608, 3072), (3072, 1), 0); del buf812  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf837, (4608, 768), (768, 1), 0), reinterpret_tensor(arg558_1, (768, 3072), (1, 768), 0), out=buf838)
        del arg558_1
        buf839 = reinterpret_tensor(buf838, (8, 576, 3072), (1769472, 3072, 1), 0); del buf838  # reuse
        # Topologically Sorted Source Nodes: [x_688], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf839, arg559_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg559_1
        buf840 = reinterpret_tensor(buf837, (4608, 768), (768, 1), 0); del buf837  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf839, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg560_1, (3072, 768), (1, 3072), 0), out=buf840)
        del arg560_1
        buf844 = reinterpret_tensor(buf813, (8, 576, 768), (442368, 768, 1), 0); del buf813  # reuse
        # Topologically Sorted Source Nodes: [mul_204, x_692, layer_norm_139], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf833, arg555_1, buf840, arg561_1, arg563_1, arg564_1, buf844, 4608, 768, grid=grid(4608), stream=stream0)
        del arg563_1
        del arg564_1
        buf845 = buf818; del buf818  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf844, (4608, 768), (768, 1), 0), reinterpret_tensor(arg565_1, (768, 2304), (1, 768), 0), out=buf845)
        del arg565_1
        buf846 = reinterpret_tensor(buf844, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf844  # reuse
        # Topologically Sorted Source Nodes: [q_69, attn_335], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf845, arg566_1, buf846, 3538944, grid=grid(3538944), stream=stream0)
        buf847 = reinterpret_tensor(buf806, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf806  # reuse
        # Topologically Sorted Source Nodes: [attn_335], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf845, arg566_1, buf847, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf848 = reinterpret_tensor(buf828, (128, 576, 576), (331776, 576, 1), 0); del buf828  # reuse
        # Topologically Sorted Source Nodes: [attn_335], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf846, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf847, (128, 48, 576), (27648, 576, 1), 0), out=buf848)
        buf849 = reinterpret_tensor(buf827, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf827  # reuse
        # Topologically Sorted Source Nodes: [linear_416], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf848, buf849, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf850 = reinterpret_tensor(buf848, (2654208, 16), (16, 1), 0); del buf848  # reuse
        # Topologically Sorted Source Nodes: [linear_416], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf849, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg567_1, (16, 16), (1, 16), 0), out=buf850)
        del arg567_1
        buf851 = buf825; del buf825  # reuse
        buf852 = buf824; del buf824  # reuse
        # Topologically Sorted Source Nodes: [attn_337], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf850, arg568_1, buf851, buf852, 73728, 576, grid=grid(73728), stream=stream0)
        buf853 = reinterpret_tensor(buf850, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf850  # reuse
        # Topologically Sorted Source Nodes: [linear_417], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf853, arg568_1, buf851, buf852, 42467328, grid=grid(42467328), stream=stream0)
        del arg568_1
        buf854 = reinterpret_tensor(buf849, (2654208, 16), (16, 1), 0); del buf849  # reuse
        # Topologically Sorted Source Nodes: [linear_417], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf853, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg569_1, (16, 16), (1, 16), 0), out=buf854)
        del arg569_1
        buf855 = reinterpret_tensor(buf853, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf853  # reuse
        # Topologically Sorted Source Nodes: [matmul_135], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf854, arg570_1, buf855, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg570_1
        buf856 = reinterpret_tensor(buf847, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf847  # reuse
        # Topologically Sorted Source Nodes: [matmul_135], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf845, arg566_1, buf856, 3538944, grid=grid(3538944), stream=stream0)
        del arg566_1
        buf857 = reinterpret_tensor(buf846, (128, 576, 48), (27648, 48, 1), 0); del buf846  # reuse
        # Topologically Sorted Source Nodes: [matmul_135], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf855, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf856, (128, 576, 48), (27648, 48, 1), 0), out=buf857)
        buf858 = reinterpret_tensor(buf856, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf856  # reuse
        # Topologically Sorted Source Nodes: [x_693], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf857, buf858, 3538944, grid=grid(3538944), stream=stream0)
        buf859 = reinterpret_tensor(buf857, (4608, 768), (768, 1), 0); del buf857  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf858, (4608, 768), (768, 1), 0), reinterpret_tensor(arg571_1, (768, 768), (1, 768), 0), out=buf859)
        del arg571_1
        buf860 = reinterpret_tensor(buf859, (8, 576, 768), (442368, 768, 1), 0); del buf859  # reuse
        buf864 = reinterpret_tensor(buf858, (8, 576, 768), (442368, 768, 1), 0); del buf858  # reuse
        # Topologically Sorted Source Nodes: [mul_204, x_692, mul_206, x_696, layer_norm_140], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf860, buf833, arg555_1, buf840, arg561_1, arg562_1, arg572_1, arg574_1, arg575_1, buf864, 4608, 768, grid=grid(4608), stream=stream0)
        del arg555_1
        del arg561_1
        del arg562_1
        del arg572_1
        del arg574_1
        del arg575_1
        buf865 = reinterpret_tensor(buf839, (4608, 3072), (3072, 1), 0); del buf839  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf864, (4608, 768), (768, 1), 0), reinterpret_tensor(arg576_1, (768, 3072), (1, 768), 0), out=buf865)
        del arg576_1
        buf866 = reinterpret_tensor(buf865, (8, 576, 3072), (1769472, 3072, 1), 0); del buf865  # reuse
        # Topologically Sorted Source Nodes: [x_698], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf866, arg577_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg577_1
        buf867 = reinterpret_tensor(buf864, (4608, 768), (768, 1), 0); del buf864  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf866, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg578_1, (3072, 768), (1, 3072), 0), out=buf867)
        del arg578_1
        buf871 = reinterpret_tensor(buf840, (8, 576, 768), (442368, 768, 1), 0); del buf840  # reuse
        # Topologically Sorted Source Nodes: [mul_207, x_702, layer_norm_141], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf860, arg573_1, buf867, arg579_1, arg581_1, arg582_1, buf871, 4608, 768, grid=grid(4608), stream=stream0)
        del arg581_1
        del arg582_1
        buf872 = buf845; del buf845  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf871, (4608, 768), (768, 1), 0), reinterpret_tensor(arg583_1, (768, 2304), (1, 768), 0), out=buf872)
        del arg583_1
        buf873 = reinterpret_tensor(buf871, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf871  # reuse
        # Topologically Sorted Source Nodes: [q_70, attn_340], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf872, arg584_1, buf873, 3538944, grid=grid(3538944), stream=stream0)
        buf874 = reinterpret_tensor(buf833, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf833  # reuse
        # Topologically Sorted Source Nodes: [attn_340], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf872, arg584_1, buf874, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf875 = reinterpret_tensor(buf855, (128, 576, 576), (331776, 576, 1), 0); del buf855  # reuse
        # Topologically Sorted Source Nodes: [attn_340], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf873, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf874, (128, 48, 576), (27648, 576, 1), 0), out=buf875)
        buf876 = reinterpret_tensor(buf854, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf854  # reuse
        # Topologically Sorted Source Nodes: [linear_422], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf875, buf876, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf877 = reinterpret_tensor(buf875, (2654208, 16), (16, 1), 0); del buf875  # reuse
        # Topologically Sorted Source Nodes: [linear_422], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf876, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg585_1, (16, 16), (1, 16), 0), out=buf877)
        del arg585_1
        buf878 = buf852; del buf852  # reuse
        buf879 = buf851; del buf851  # reuse
        # Topologically Sorted Source Nodes: [attn_342], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf877, arg586_1, buf878, buf879, 73728, 576, grid=grid(73728), stream=stream0)
        buf880 = reinterpret_tensor(buf877, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf877  # reuse
        # Topologically Sorted Source Nodes: [linear_423], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf880, arg586_1, buf878, buf879, 42467328, grid=grid(42467328), stream=stream0)
        del arg586_1
        buf881 = reinterpret_tensor(buf876, (2654208, 16), (16, 1), 0); del buf876  # reuse
        # Topologically Sorted Source Nodes: [linear_423], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf880, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg587_1, (16, 16), (1, 16), 0), out=buf881)
        del arg587_1
        buf882 = reinterpret_tensor(buf880, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf880  # reuse
        # Topologically Sorted Source Nodes: [matmul_137], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf881, arg588_1, buf882, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg588_1
        buf883 = reinterpret_tensor(buf874, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf874  # reuse
        # Topologically Sorted Source Nodes: [matmul_137], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf872, arg584_1, buf883, 3538944, grid=grid(3538944), stream=stream0)
        del arg584_1
        buf884 = reinterpret_tensor(buf873, (128, 576, 48), (27648, 48, 1), 0); del buf873  # reuse
        # Topologically Sorted Source Nodes: [matmul_137], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf882, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf883, (128, 576, 48), (27648, 48, 1), 0), out=buf884)
        buf885 = reinterpret_tensor(buf883, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf883  # reuse
        # Topologically Sorted Source Nodes: [x_703], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf884, buf885, 3538944, grid=grid(3538944), stream=stream0)
        buf886 = reinterpret_tensor(buf884, (4608, 768), (768, 1), 0); del buf884  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf885, (4608, 768), (768, 1), 0), reinterpret_tensor(arg589_1, (768, 768), (1, 768), 0), out=buf886)
        del arg589_1
        buf887 = reinterpret_tensor(buf886, (8, 576, 768), (442368, 768, 1), 0); del buf886  # reuse
        buf891 = reinterpret_tensor(buf885, (8, 576, 768), (442368, 768, 1), 0); del buf885  # reuse
        # Topologically Sorted Source Nodes: [mul_207, x_702, mul_209, x_706, layer_norm_142], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf887, buf860, arg573_1, buf867, arg579_1, arg580_1, arg590_1, arg592_1, arg593_1, buf891, 4608, 768, grid=grid(4608), stream=stream0)
        del arg573_1
        del arg579_1
        del arg580_1
        del arg590_1
        del arg592_1
        del arg593_1
        buf892 = reinterpret_tensor(buf866, (4608, 3072), (3072, 1), 0); del buf866  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf891, (4608, 768), (768, 1), 0), reinterpret_tensor(arg594_1, (768, 3072), (1, 768), 0), out=buf892)
        del arg594_1
        buf893 = reinterpret_tensor(buf892, (8, 576, 3072), (1769472, 3072, 1), 0); del buf892  # reuse
        # Topologically Sorted Source Nodes: [x_708], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf893, arg595_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg595_1
        buf894 = reinterpret_tensor(buf891, (4608, 768), (768, 1), 0); del buf891  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf893, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg596_1, (3072, 768), (1, 3072), 0), out=buf894)
        del arg596_1
        buf898 = reinterpret_tensor(buf867, (8, 576, 768), (442368, 768, 1), 0); del buf867  # reuse
        # Topologically Sorted Source Nodes: [mul_210, x_712, layer_norm_143], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf887, arg591_1, buf894, arg597_1, arg599_1, arg600_1, buf898, 4608, 768, grid=grid(4608), stream=stream0)
        del arg599_1
        del arg600_1
        buf899 = buf872; del buf872  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf898, (4608, 768), (768, 1), 0), reinterpret_tensor(arg601_1, (768, 2304), (1, 768), 0), out=buf899)
        del arg601_1
        buf900 = reinterpret_tensor(buf898, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf898  # reuse
        # Topologically Sorted Source Nodes: [q_71, attn_345], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf899, arg602_1, buf900, 3538944, grid=grid(3538944), stream=stream0)
        buf901 = reinterpret_tensor(buf860, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf860  # reuse
        # Topologically Sorted Source Nodes: [attn_345], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf899, arg602_1, buf901, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf902 = reinterpret_tensor(buf882, (128, 576, 576), (331776, 576, 1), 0); del buf882  # reuse
        # Topologically Sorted Source Nodes: [attn_345], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf900, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf901, (128, 48, 576), (27648, 576, 1), 0), out=buf902)
        buf903 = reinterpret_tensor(buf881, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf881  # reuse
        # Topologically Sorted Source Nodes: [linear_428], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf902, buf903, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf904 = reinterpret_tensor(buf902, (2654208, 16), (16, 1), 0); del buf902  # reuse
        # Topologically Sorted Source Nodes: [linear_428], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf903, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg603_1, (16, 16), (1, 16), 0), out=buf904)
        del arg603_1
        buf905 = buf879; del buf879  # reuse
        buf906 = buf878; del buf878  # reuse
        # Topologically Sorted Source Nodes: [attn_347], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf904, arg604_1, buf905, buf906, 73728, 576, grid=grid(73728), stream=stream0)
        buf907 = reinterpret_tensor(buf904, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf904  # reuse
        # Topologically Sorted Source Nodes: [linear_429], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf907, arg604_1, buf905, buf906, 42467328, grid=grid(42467328), stream=stream0)
        del arg604_1
        buf908 = reinterpret_tensor(buf903, (2654208, 16), (16, 1), 0); del buf903  # reuse
        # Topologically Sorted Source Nodes: [linear_429], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf907, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg605_1, (16, 16), (1, 16), 0), out=buf908)
        del arg605_1
        buf909 = reinterpret_tensor(buf907, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf907  # reuse
        # Topologically Sorted Source Nodes: [matmul_139], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf908, arg606_1, buf909, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg606_1
        buf910 = reinterpret_tensor(buf901, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf901  # reuse
        # Topologically Sorted Source Nodes: [matmul_139], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf899, arg602_1, buf910, 3538944, grid=grid(3538944), stream=stream0)
        del arg602_1
        buf911 = reinterpret_tensor(buf900, (128, 576, 48), (27648, 48, 1), 0); del buf900  # reuse
        # Topologically Sorted Source Nodes: [matmul_139], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf909, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf910, (128, 576, 48), (27648, 48, 1), 0), out=buf911)
        buf912 = reinterpret_tensor(buf910, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf910  # reuse
        # Topologically Sorted Source Nodes: [x_713], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf911, buf912, 3538944, grid=grid(3538944), stream=stream0)
        buf913 = reinterpret_tensor(buf911, (4608, 768), (768, 1), 0); del buf911  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf912, (4608, 768), (768, 1), 0), reinterpret_tensor(arg607_1, (768, 768), (1, 768), 0), out=buf913)
        del arg607_1
        buf914 = reinterpret_tensor(buf913, (8, 576, 768), (442368, 768, 1), 0); del buf913  # reuse
        buf918 = reinterpret_tensor(buf912, (8, 576, 768), (442368, 768, 1), 0); del buf912  # reuse
        # Topologically Sorted Source Nodes: [mul_210, x_712, mul_212, x_716, layer_norm_144], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf914, buf887, arg591_1, buf894, arg597_1, arg598_1, arg608_1, arg610_1, arg611_1, buf918, 4608, 768, grid=grid(4608), stream=stream0)
        del arg591_1
        del arg597_1
        del arg598_1
        del arg608_1
        del arg610_1
        del arg611_1
        buf919 = reinterpret_tensor(buf893, (4608, 3072), (3072, 1), 0); del buf893  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf918, (4608, 768), (768, 1), 0), reinterpret_tensor(arg612_1, (768, 3072), (1, 768), 0), out=buf919)
        del arg612_1
        buf920 = reinterpret_tensor(buf919, (8, 576, 3072), (1769472, 3072, 1), 0); del buf919  # reuse
        # Topologically Sorted Source Nodes: [x_718], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf920, arg613_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg613_1
        buf921 = reinterpret_tensor(buf918, (4608, 768), (768, 1), 0); del buf918  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf920, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg614_1, (3072, 768), (1, 3072), 0), out=buf921)
        del arg614_1
        buf925 = reinterpret_tensor(buf894, (8, 576, 768), (442368, 768, 1), 0); del buf894  # reuse
        # Topologically Sorted Source Nodes: [mul_213, x_722, layer_norm_145], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf914, arg609_1, buf921, arg615_1, arg617_1, arg618_1, buf925, 4608, 768, grid=grid(4608), stream=stream0)
        del arg617_1
        del arg618_1
        buf926 = buf899; del buf899  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf925, (4608, 768), (768, 1), 0), reinterpret_tensor(arg619_1, (768, 2304), (1, 768), 0), out=buf926)
        del arg619_1
        buf927 = reinterpret_tensor(buf925, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf925  # reuse
        # Topologically Sorted Source Nodes: [q_72, attn_350], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf926, arg620_1, buf927, 3538944, grid=grid(3538944), stream=stream0)
        buf928 = reinterpret_tensor(buf887, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf887  # reuse
        # Topologically Sorted Source Nodes: [attn_350], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf926, arg620_1, buf928, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf929 = reinterpret_tensor(buf909, (128, 576, 576), (331776, 576, 1), 0); del buf909  # reuse
        # Topologically Sorted Source Nodes: [attn_350], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf927, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf928, (128, 48, 576), (27648, 576, 1), 0), out=buf929)
        buf930 = reinterpret_tensor(buf908, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf908  # reuse
        # Topologically Sorted Source Nodes: [linear_434], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf929, buf930, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf931 = reinterpret_tensor(buf929, (2654208, 16), (16, 1), 0); del buf929  # reuse
        # Topologically Sorted Source Nodes: [linear_434], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf930, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg621_1, (16, 16), (1, 16), 0), out=buf931)
        del arg621_1
        buf932 = buf906; del buf906  # reuse
        buf933 = buf905; del buf905  # reuse
        # Topologically Sorted Source Nodes: [attn_352], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf931, arg622_1, buf932, buf933, 73728, 576, grid=grid(73728), stream=stream0)
        buf934 = reinterpret_tensor(buf931, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf931  # reuse
        # Topologically Sorted Source Nodes: [linear_435], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf934, arg622_1, buf932, buf933, 42467328, grid=grid(42467328), stream=stream0)
        del arg622_1
        buf935 = reinterpret_tensor(buf930, (2654208, 16), (16, 1), 0); del buf930  # reuse
        # Topologically Sorted Source Nodes: [linear_435], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf934, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg623_1, (16, 16), (1, 16), 0), out=buf935)
        del arg623_1
        buf936 = reinterpret_tensor(buf934, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf934  # reuse
        # Topologically Sorted Source Nodes: [matmul_141], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf935, arg624_1, buf936, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg624_1
        buf937 = reinterpret_tensor(buf928, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf928  # reuse
        # Topologically Sorted Source Nodes: [matmul_141], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf926, arg620_1, buf937, 3538944, grid=grid(3538944), stream=stream0)
        del arg620_1
        buf938 = reinterpret_tensor(buf927, (128, 576, 48), (27648, 48, 1), 0); del buf927  # reuse
        # Topologically Sorted Source Nodes: [matmul_141], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf936, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf937, (128, 576, 48), (27648, 48, 1), 0), out=buf938)
        buf939 = reinterpret_tensor(buf937, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf937  # reuse
        # Topologically Sorted Source Nodes: [x_723], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf938, buf939, 3538944, grid=grid(3538944), stream=stream0)
        buf940 = reinterpret_tensor(buf938, (4608, 768), (768, 1), 0); del buf938  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf939, (4608, 768), (768, 1), 0), reinterpret_tensor(arg625_1, (768, 768), (1, 768), 0), out=buf940)
        del arg625_1
        buf941 = reinterpret_tensor(buf940, (8, 576, 768), (442368, 768, 1), 0); del buf940  # reuse
        buf945 = reinterpret_tensor(buf939, (8, 576, 768), (442368, 768, 1), 0); del buf939  # reuse
        # Topologically Sorted Source Nodes: [mul_213, x_722, mul_215, x_726, layer_norm_146], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf941, buf914, arg609_1, buf921, arg615_1, arg616_1, arg626_1, arg628_1, arg629_1, buf945, 4608, 768, grid=grid(4608), stream=stream0)
        del arg609_1
        del arg615_1
        del arg616_1
        del arg626_1
        del arg628_1
        del arg629_1
        buf946 = reinterpret_tensor(buf920, (4608, 3072), (3072, 1), 0); del buf920  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf945, (4608, 768), (768, 1), 0), reinterpret_tensor(arg630_1, (768, 3072), (1, 768), 0), out=buf946)
        del arg630_1
        buf947 = reinterpret_tensor(buf946, (8, 576, 3072), (1769472, 3072, 1), 0); del buf946  # reuse
        # Topologically Sorted Source Nodes: [x_728], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf947, arg631_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg631_1
        buf948 = reinterpret_tensor(buf945, (4608, 768), (768, 1), 0); del buf945  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf947, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg632_1, (3072, 768), (1, 3072), 0), out=buf948)
        del arg632_1
        buf952 = reinterpret_tensor(buf921, (8, 576, 768), (442368, 768, 1), 0); del buf921  # reuse
        # Topologically Sorted Source Nodes: [mul_216, x_732, layer_norm_147], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_13.run(buf941, arg627_1, buf948, arg633_1, arg635_1, arg636_1, buf952, 4608, 768, grid=grid(4608), stream=stream0)
        del arg635_1
        del arg636_1
        buf953 = buf926; del buf926  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf952, (4608, 768), (768, 1), 0), reinterpret_tensor(arg637_1, (768, 2304), (1, 768), 0), out=buf953)
        del arg637_1
        buf954 = reinterpret_tensor(buf952, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf952  # reuse
        # Topologically Sorted Source Nodes: [q_73, attn_355], Original ATen: [aten.mul, aten.clone]
        triton_poi_fused_clone_mul_3.run(buf953, arg638_1, buf954, 3538944, grid=grid(3538944), stream=stream0)
        buf955 = reinterpret_tensor(buf914, (8, 16, 48, 576), (442368, 27648, 576, 1), 0); del buf914  # reuse
        # Topologically Sorted Source Nodes: [attn_355], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf953, arg638_1, buf955, 6144, 576, grid=grid(6144, 576), stream=stream0)
        buf956 = reinterpret_tensor(buf936, (128, 576, 576), (331776, 576, 1), 0); del buf936  # reuse
        # Topologically Sorted Source Nodes: [attn_355], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf954, (128, 576, 48), (27648, 48, 1), 0), reinterpret_tensor(buf955, (128, 48, 576), (27648, 576, 1), 0), out=buf956)
        buf957 = reinterpret_tensor(buf935, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf935  # reuse
        # Topologically Sorted Source Nodes: [linear_440], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf956, buf957, 2654208, 16, grid=grid(2654208, 16), stream=stream0)
        buf958 = reinterpret_tensor(buf956, (2654208, 16), (16, 1), 0); del buf956  # reuse
        # Topologically Sorted Source Nodes: [linear_440], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf957, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg639_1, (16, 16), (1, 16), 0), out=buf958)
        del arg639_1
        buf959 = buf933; del buf933  # reuse
        buf960 = buf932; del buf932  # reuse
        # Topologically Sorted Source Nodes: [attn_357], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf958, arg640_1, buf959, buf960, 73728, 576, grid=grid(73728), stream=stream0)
        buf961 = reinterpret_tensor(buf958, (8, 576, 576, 16), (5308416, 9216, 16, 1), 0); del buf958  # reuse
        # Topologically Sorted Source Nodes: [linear_441], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf961, arg640_1, buf959, buf960, 42467328, grid=grid(42467328), stream=stream0)
        del arg640_1
        del buf959
        del buf960
        buf962 = reinterpret_tensor(buf957, (2654208, 16), (16, 1), 0); del buf957  # reuse
        # Topologically Sorted Source Nodes: [linear_441], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf961, (2654208, 16), (16, 1), 0), reinterpret_tensor(arg641_1, (16, 16), (1, 16), 0), out=buf962)
        del arg641_1
        buf963 = reinterpret_tensor(buf961, (8, 16, 576, 576), (5308416, 331776, 576, 1), 0); del buf961  # reuse
        # Topologically Sorted Source Nodes: [matmul_143], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf962, arg642_1, buf963, 128, 331776, grid=grid(128, 331776), stream=stream0)
        del arg642_1
        del buf962
        buf964 = reinterpret_tensor(buf955, (8, 16, 576, 48), (442368, 27648, 48, 1), 0); del buf955  # reuse
        # Topologically Sorted Source Nodes: [matmul_143], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf953, arg638_1, buf964, 3538944, grid=grid(3538944), stream=stream0)
        del arg638_1
        del buf953
        buf965 = reinterpret_tensor(buf954, (128, 576, 48), (27648, 48, 1), 0); del buf954  # reuse
        # Topologically Sorted Source Nodes: [matmul_143], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf963, (128, 576, 576), (331776, 576, 1), 0), reinterpret_tensor(buf964, (128, 576, 48), (27648, 48, 1), 0), out=buf965)
        del buf963
        buf966 = reinterpret_tensor(buf964, (8, 576, 16, 48), (442368, 768, 48, 1), 0); del buf964  # reuse
        # Topologically Sorted Source Nodes: [x_733], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf965, buf966, 3538944, grid=grid(3538944), stream=stream0)
        buf967 = reinterpret_tensor(buf965, (4608, 768), (768, 1), 0); del buf965  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf966, (4608, 768), (768, 1), 0), reinterpret_tensor(arg643_1, (768, 768), (1, 768), 0), out=buf967)
        del arg643_1
        buf968 = reinterpret_tensor(buf967, (8, 576, 768), (442368, 768, 1), 0); del buf967  # reuse
        buf972 = reinterpret_tensor(buf966, (8, 576, 768), (442368, 768, 1), 0); del buf966  # reuse
        # Topologically Sorted Source Nodes: [mul_216, x_732, mul_218, x_736, layer_norm_148], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_14.run(buf968, buf941, arg627_1, buf948, arg633_1, arg634_1, arg644_1, arg646_1, arg647_1, buf972, 4608, 768, grid=grid(4608), stream=stream0)
        del arg627_1
        del arg633_1
        del arg634_1
        del arg644_1
        del arg646_1
        del arg647_1
        del buf941
        del buf948
        buf973 = reinterpret_tensor(buf947, (4608, 3072), (3072, 1), 0); del buf947  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf972, (4608, 768), (768, 1), 0), reinterpret_tensor(arg648_1, (768, 3072), (1, 768), 0), out=buf973)
        del arg648_1
        buf974 = reinterpret_tensor(buf973, (8, 576, 3072), (1769472, 3072, 1), 0); del buf973  # reuse
        # Topologically Sorted Source Nodes: [x_738], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_12.run(buf974, arg649_1, 14155776, grid=grid(14155776), stream=stream0)
        del arg649_1
        buf975 = reinterpret_tensor(buf972, (4608, 768), (768, 1), 0); del buf972  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf974, (4608, 3072), (3072, 1), 0), reinterpret_tensor(arg650_1, (3072, 768), (1, 3072), 0), out=buf975)
        del arg650_1
        del buf974
        buf976 = empty_strided_cuda((8, 577, 768), (443136, 768, 1), torch.float32)
        buf980 = empty_strided_cuda((8, 577, 768), (443136, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_2, layer_norm_149], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_15.run(arg652_1, buf968, arg645_1, buf975, arg651_1, arg654_1, arg655_1, buf976, buf980, 4616, 768, grid=grid(4616), stream=stream0)
        del arg654_1
        del arg655_1
        buf981 = empty_strided_cuda((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_445], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg657_1, reinterpret_tensor(buf980, (8, 768), (443136, 1), 0), reinterpret_tensor(arg656_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf981)
        del arg656_1
        del arg657_1
        buf982 = reinterpret_tensor(buf976, (4616, 768), (768, 1), 0); del buf976  # reuse
        # Topologically Sorted Source Nodes: [linear_446], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg659_1, reinterpret_tensor(buf980, (4616, 768), (768, 1), 0), reinterpret_tensor(arg658_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf982)
        del arg658_1
        del arg659_1
        buf983 = empty_strided_cuda((4616, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_447], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg661_1, reinterpret_tensor(buf980, (4616, 768), (768, 1), 0), reinterpret_tensor(arg660_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf983)
        del arg660_1
        del arg661_1
        # Topologically Sorted Source Nodes: [x_cls_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf984 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf981, (8, 16, 1, 48), (768, 48, 768, 1), 0), reinterpret_tensor(buf982, (8, 16, 577, 48), (443136, 48, 768, 1), 0), reinterpret_tensor(buf983, (8, 16, 577, 48), (443136, 48, 768, 1), 0), None, False)
        buf985 = buf984[0]
        del buf984
        buf989 = buf981; del buf981  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf985, (8, 768), (768, 1), 0), reinterpret_tensor(arg662_1, (768, 768), (1, 768), 0), out=buf989)
        del arg662_1
        buf993 = reinterpret_tensor(buf985, (8, 1, 768), (768, 768, 1), 0); del buf985  # reuse
        # Topologically Sorted Source Nodes: [mul_220, x_cls_16, layer_norm_150], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_16.run(arg652_1, arg653_1, buf989, arg663_1, arg665_1, arg666_1, buf993, 8, 768, grid=grid(8), stream=stream0)
        del arg665_1
        del arg666_1
        buf994 = empty_strided_cuda((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf993, (8, 768), (768, 1), 0), reinterpret_tensor(arg667_1, (768, 3072), (1, 768), 0), out=buf994)
        del arg667_1
        buf995 = reinterpret_tensor(buf994, (8, 1, 3072), (3072, 3072, 1), 0); del buf994  # reuse
        # Topologically Sorted Source Nodes: [x_744], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf995, arg668_1, 24576, grid=grid(24576), stream=stream0)
        del arg668_1
        buf996 = reinterpret_tensor(buf993, (8, 768), (768, 1), 0); del buf993  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf995, (8, 3072), (3072, 1), 0), reinterpret_tensor(arg669_1, (3072, 768), (1, 3072), 0), out=buf996)
        del arg669_1
        buf999 = reinterpret_tensor(buf983, (8, 577, 768), (443136, 768, 1), 0); del buf983  # reuse
        buf997 = reinterpret_tensor(buf999, (8, 1, 768), (443136, 768, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [mul_220, x_cls_16, mul_221, x_cls_17], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_18.run(arg652_1, arg653_1, buf989, arg663_1, arg664_1, buf996, arg670_1, buf997, 6144, grid=grid(6144), stream=stream0)
        del arg652_1
        del arg653_1
        del arg663_1
        del arg664_1
        del arg670_1
        del buf989
        buf998 = reinterpret_tensor(buf999, (8, 576, 768), (443136, 768, 1), 768)  # alias
        # Topologically Sorted Source Nodes: [mul_219, x_742], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_19.run(buf968, arg645_1, buf975, arg651_1, buf998, 3538944, grid=grid(3538944), stream=stream0)
        del arg645_1
        del arg651_1
        del buf968
        del buf975
        buf1003 = reinterpret_tensor(buf982, (8, 577, 768), (443136, 768, 1), 0); del buf982  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_151], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_20.run(buf999, arg672_1, arg673_1, buf1003, 4616, 768, grid=grid(4616), stream=stream0)
        del arg672_1
        del arg673_1
        buf1004 = buf996; del buf996  # reuse
        # Topologically Sorted Source Nodes: [linear_451], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg675_1, reinterpret_tensor(buf1003, (8, 768), (443136, 1), 0), reinterpret_tensor(arg674_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1004)
        del arg674_1
        del arg675_1
        buf1005 = reinterpret_tensor(buf980, (4616, 768), (768, 1), 0); del buf980  # reuse
        # Topologically Sorted Source Nodes: [linear_452], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg677_1, reinterpret_tensor(buf1003, (4616, 768), (768, 1), 0), reinterpret_tensor(arg676_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1005)
        del arg676_1
        del arg677_1
        buf1006 = empty_strided_cuda((4616, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_453], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg679_1, reinterpret_tensor(buf1003, (4616, 768), (768, 1), 0), reinterpret_tensor(arg678_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf1006)
        del arg678_1
        del arg679_1
        del buf1003
        # Topologically Sorted Source Nodes: [x_cls_18], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf1007 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf1004, (8, 16, 1, 48), (768, 48, 768, 1), 0), reinterpret_tensor(buf1005, (8, 16, 577, 48), (443136, 48, 768, 1), 0), reinterpret_tensor(buf1006, (8, 16, 577, 48), (443136, 48, 768, 1), 0), None, False)
        del buf1005
        buf1008 = buf1007[0]
        del buf1007
        buf1012 = buf1004; del buf1004  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1008, (8, 768), (768, 1), 0), reinterpret_tensor(arg680_1, (768, 768), (1, 768), 0), out=buf1012)
        del arg680_1
        buf1016 = reinterpret_tensor(buf1008, (8, 1, 768), (768, 768, 1), 0); del buf1008  # reuse
        # Topologically Sorted Source Nodes: [mul_222, x_cls_22, layer_norm_152], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_per_fused_add_mul_native_layer_norm_21.run(buf997, arg671_1, buf1012, arg681_1, arg683_1, arg684_1, buf1016, 8, 768, grid=grid(8), stream=stream0)
        del arg683_1
        del arg684_1
        buf1017 = reinterpret_tensor(buf995, (8, 3072), (3072, 1), 0); del buf995  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1016, (8, 768), (768, 1), 0), reinterpret_tensor(arg685_1, (768, 3072), (1, 768), 0), out=buf1017)
        del arg685_1
        buf1018 = reinterpret_tensor(buf1017, (8, 1, 3072), (3072, 3072, 1), 0); del buf1017  # reuse
        # Topologically Sorted Source Nodes: [x_749], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_17.run(buf1018, arg686_1, 24576, grid=grid(24576), stream=stream0)
        del arg686_1
        buf1019 = reinterpret_tensor(buf1016, (8, 768), (768, 1), 0); del buf1016  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1018, (8, 3072), (3072, 1), 0), reinterpret_tensor(arg687_1, (3072, 768), (1, 3072), 0), out=buf1019)
        del arg687_1
        del buf1018
        buf1020 = reinterpret_tensor(buf1006, (8, 577, 768), (443136, 768, 1), 0); del buf1006  # reuse
        buf1021 = empty_strided_cuda((8, 577, 1), (577, 1, 4640), torch.float32)
        buf1022 = empty_strided_cuda((8, 577, 1), (577, 1, 4640), torch.float32)
        # Topologically Sorted Source Nodes: [x_753, x_754], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_22.run(buf997, arg671_1, buf1012, arg681_1, arg682_1, buf1019, arg688_1, buf998, buf1020, buf1021, buf1022, 4616, 768, grid=grid(4616), stream=stream0)
        del arg671_1
        del arg681_1
        del arg682_1
        del arg688_1
        del buf1012
        del buf997
        del buf998
        del buf999
        buf1024 = buf1019; del buf1019  # reuse
        # Topologically Sorted Source Nodes: [x_756], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf1020, buf1021, buf1022, arg689_1, arg690_1, buf1024, 6144, grid=grid(6144), stream=stream0)
        del arg689_1
        del arg690_1
        del buf1020
        del buf1021
        del buf1022
        buf1025 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_756, x_757], Original ATen: [aten.clone, aten.addmm]
        extern_kernels.addmm(arg692_1, buf1024, reinterpret_tensor(arg691_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf1025)
        del arg691_1
        del arg692_1
        del buf1024
    return (buf1025, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 384, 384), (442368, 147456, 384, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 576, 768), (442368, 768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('cait_m36_384', benchmark_compiled_module)
