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


# kernel path: /tmp/torchinductor_sahanp/on/con6rjgwhqhmjv4btt6r4kj3fx2vjsmxjqdnk3bu4gljzkyzm6c4.py
# Topologically Sorted Source Nodes: [x_155, x_157], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_155 => add_117
#   x_157 => clone_159, var_mean_25
# Graph fragment:
#   %add_117 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_126, %arg3_1), kwargs = {})
#   %clone_159 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_117,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_159, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_0 = async_compile.triton('triton_red_fused_add_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 6
    x1 = (xindex // 6) % 196
    x2 = (xindex // 1176)
    x5 = xindex % 1176
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (150528*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x5 + (1184*x2)), tmp6, xmask)
    tl.store(out_ptr1 + (x5 + (1184*x2)), tmp7, xmask)
    tl.store(out_ptr2 + (x5 + (1184*x2)), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rq/crqp423dvihdwjgc7juawzwyhydp5sni3lu6hmmn6v55yyxaey33.py
# Topologically Sorted Source Nodes: [x_155, x_157], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_155 => add_117
#   x_157 => clone_159, var_mean_25
# Graph fragment:
#   %add_117 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_126, %arg3_1), kwargs = {})
#   %clone_159 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_117,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_159, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_1 = async_compile.triton('triton_per_fused_add_native_layer_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
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


# kernel path: /tmp/torchinductor_sahanp/ys/cyssinhxscp2owjq4hvepsluffstmcfuszwnx3uit2j53jlhwen2.py
# Topologically Sorted Source Nodes: [x_155, x_157], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_155 => add_117
#   x_157 => add_118, add_119, clone_159, mul_118, mul_119, rsqrt_25, sub_67, var_mean_25
# Graph fragment:
#   %add_117 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_126, %arg3_1), kwargs = {})
#   %clone_159 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_117,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_159, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_159, %getitem_57), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_56, 1e-06), kwargs = {})
#   %rsqrt_25 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_118,), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %rsqrt_25), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %arg5_1), kwargs = {})
#   %add_119 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %arg6_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_2 = async_compile.triton('triton_poi_fused_add_native_layer_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_sahanp/op/copkfv43dyjcbvn2mdxo7tkzjj6fktx2g3o6eolqaqfp4bwxt6fi.py
# Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_24 => clone_161
# Graph fragment:
#   %clone_161 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_81,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_3 = async_compile.triton('triton_poi_fused_clone_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 48
    x1 = (xindex // 48) % 196
    x2 = (xindex // 9408) % 16
    x3 = (xindex // 150528)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (1536*x1) + (301056*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tt/cttqtu3qnbhbncc4zkvn7kvqgpfmkuy7fjw2hfldu2eq7auimdp5.py
# Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_24 => clone_162
# Graph fragment:
#   %clone_162 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_82,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (768 + y0 + (1536*x2) + (301056*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h5/ch5ggth5mqy56sxdn65gg5uxdvwstwmpbfutf26rgvan5wmsph4x.py
# Topologically Sorted Source Nodes: [patch_score_21], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   patch_score_21 => exp_22, sum_33
# Graph fragment:
#   %mul_tensor_22 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_301, 1), kwargs = {})
#   %amax_default_11 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_22, [-1], True), kwargs = {})
#   %sub_tensor_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_22, %amax_default_11), kwargs = {})
#   %mul_tensor_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_11, 0.14433756729740643), kwargs = {})
#   %exp_22 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_23,), kwargs = {})
#   %sum_33 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_22, [-1], True), kwargs = {})
triton_per_fused__softmax_5 = async_compile.triton('triton_per_fused__softmax_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_5(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = 0.14433756729740643
    tmp9 = tmp7 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7x/c7xhaz2v62bshzv3gjuftsj3yr3anobn4b5pzufz4bdv7nfdgmdu.py
# Topologically Sorted Source Nodes: [rel_indices, setitem, setitem_1, setitem_2, to], Original ATen: [aten.zeros, aten.copy, aten._to_copy]
# Source node to ATen node mapping:
#   rel_indices => full_default
#   setitem => copy
#   setitem_1 => copy_1
#   setitem_2 => copy_2
#   to => device_put
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 196, 196, 3], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select, %unsqueeze_2), kwargs = {})
#   %select_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %copy, 3, 2), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_3, %unsqueeze_3), kwargs = {})
#   %select_scatter_default_1 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %copy_1, 3, 1), kwargs = {})
#   %copy_2 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_6, %unsqueeze_4), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %copy_2, 3, 0), kwargs = {})
#   %device_put : [num_users=2] = call_function[target=torch.ops.prims.device_put.default](args = (%select_scatter_default_2, cuda:0), kwargs = {})
triton_poi_fused__to_copy_copy_zeros_6 = async_compile.triton('triton_poi_fused__to_copy_copy_zeros_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy_zeros_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_copy_zeros_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 115248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 2, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = ((x1 // 14)*(x1 // 14)) + ((x2 // 14)*(x2 // 14)) + ((x1 % 14)*(x1 % 14)) + ((x2 % 14)*(x2 % 14)) + ((-2)*(x1 // 14)*(x2 // 14)) + ((-2)*(x1 % 14)*(x2 % 14))
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 0.0
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tmp11 = ((-1)*(x2 // 14)) + (x1 // 14)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.where(tmp4, tmp12, tmp10)
    tmp14 = ((-1)*(x2 % 14)) + (x1 % 14)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.where(tmp2, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7u/c7umnxs6ipcwxuruod5ndmsal5l4acug4khcg2vjoy56razlmt6u.py
# Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   linear_70 => clone_160
# Graph fragment:
#   %clone_160 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_80,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 921984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 115248
    x1 = (xindex // 115248)
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0 + (115264*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ld/cld5ha4gudv33gt2gbhjphvni32mxlsopyqvidriga6guuf6qehr.py
# Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   linear_70 => mm_33
# Graph fragment:
#   %mm_33 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_297, %permute_129), kwargs = {})
triton_poi_fused_mm_8 = async_compile.triton('triton_poi_fused_mm_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 921984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = (xindex // 3)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3*(x1 % 38416)) + (115264*(x1 // 38416))), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tu/ctua3eevn5q3cz4yihik6ci7xxjti552qwmztthvj4su2bwhpxfd.py
# Topologically Sorted Source Nodes: [pos_score_32], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   pos_score_32 => amax_23, clone_163, exp_23, sub_69, sum_34
# Graph fragment:
#   %clone_163 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_130,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_23 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_163, [-1], True), kwargs = {})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_163, %amax_23), kwargs = {})
#   %exp_23 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_69,), kwargs = {})
#   %sum_34 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_23, [-1], True), kwargs = {})
triton_red_fused__softmax_9 = async_compile.triton('triton_red_fused__softmax_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_9(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (3136*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp6 = tl.load(in_ptr0 + (x0 + (16*r2) + (3136*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp6 + tmp1
        tmp8 = tmp7 - tmp4
        tmp9 = tl_math.exp(tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/in/cinn3mw6wci3qxtdhpayi7b6shhkttqibpec6ivmomdlo3xruykm.py
# Topologically Sorted Source Nodes: [sigmoid_20, sub_20, patch_score_21, mul_33, sigmoid_21, pos_score_32, mul_34, attn_36], Original ATen: [aten.sigmoid, aten.rsub, aten._softmax, aten.mul, aten.add]
# Source node to ATen node mapping:
#   attn_36 => add_121
#   mul_33 => mul_121
#   mul_34 => mul_122
#   patch_score_21 => div_32, exp_22
#   pos_score_32 => clone_163, div_33, exp_23, sub_69
#   sigmoid_20 => sigmoid_20
#   sigmoid_21 => sigmoid_21
#   sub_20 => sub_70
# Graph fragment:
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_302,), kwargs = {})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sigmoid_20), kwargs = {})
#   %mul_tensor_22 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_301, 1), kwargs = {})
#   %sub_tensor_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_22, %amax_default_11), kwargs = {})
#   %mul_tensor_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_11, 0.14433756729740643), kwargs = {})
#   %exp_22 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_23,), kwargs = {})
#   %div_32 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_22, %sum_33), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %div_32), kwargs = {})
#   %sigmoid_21 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_302,), kwargs = {})
#   %clone_163 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_130,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_163, %amax_23), kwargs = {})
#   %exp_23 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_69,), kwargs = {})
#   %div_33 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_23, %sum_34), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_21, %div_33), kwargs = {})
#   %add_121 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_121, %mul_122), kwargs = {})
triton_poi_fused__softmax_add_mul_rsub_sigmoid_10 = async_compile.triton('triton_poi_fused__softmax_add_mul_rsub_sigmoid_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[128, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_mul_rsub_sigmoid_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_mul_rsub_sigmoid_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 38416
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 16
    x5 = xindex
    y4 = yindex
    x3 = (xindex // 196)
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x5 + (38416*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3 + (196*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x3 + (196*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0 + (16*x5) + (614656*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0 + (16*x3) + (3136*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (y0 + (16*x3) + (3136*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp5 = tmp4 * tmp2
    tmp7 = tmp5 - tmp6
    tmp8 = 0.14433756729740643
    tmp9 = tmp7 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp12 = tmp10 / tmp11
    tmp13 = tmp3 * tmp12
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl_math.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp1 * tmp21
    tmp23 = tmp13 + tmp22
    tl.store(out_ptr0 + (x5 + (38432*y4)), tmp23, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dn/cdn5bv445nndh6gwrq3xg5qez6txl26a73ca2ltxyg2izzxuizkm.py
# Topologically Sorted Source Nodes: [sum_11, attn_37], Original ATen: [aten.sum, aten.div]
# Source node to ATen node mapping:
#   attn_37 => div_34
#   sum_11 => sum_35
# Graph fragment:
#   %sum_35 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_121, [-1]), kwargs = {})
#   %div_34 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_121, %unsqueeze_60), kwargs = {})
triton_red_fused_div_sum_11 = async_compile.triton('triton_red_fused_div_sum_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_sum_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_sum_11(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (196*x0) + (38432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp4 = tl.load(in_ptr0 + (r2 + (196*x0) + (38432*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 / tmp2
        tl.store(out_ptr1 + (r2 + (196*x0) + (38432*x1)), tmp5, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gv/cgv7zyz47juhryahrz2mjc3itbulf52gjrcvcpmi27cn5bsvjn5s.py
# Topologically Sorted Source Nodes: [matmul_25], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_25 => clone_165
# Graph fragment:
#   %clone_165 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_84,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_12 = async_compile.triton('triton_poi_fused_clone_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 48
    x1 = (xindex // 48) % 196
    x2 = (xindex // 9408) % 16
    x3 = (xindex // 150528)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (768*x1) + (150528*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ad/cadrhx7efwo55wi6nrsxrfsebs3ffzbuj5noqhwgncxg4fkt7wps.py
# Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_158 => clone_166
# Graph fragment:
#   %clone_166 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_134,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_13 = async_compile.triton('triton_poi_fused_clone_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 48
    x1 = (xindex // 48) % 16
    x2 = (xindex // 768) % 196
    x3 = (xindex // 150528)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (9408*x1) + (150528*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/l5/cl5k5lnrunuxcjh4jr3tak5c2mlqezhhax7purju5ozbbd6q37yj.py
# Topologically Sorted Source Nodes: [x_155, x_161, x_162], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_155 => add_117
#   x_161 => add_122
#   x_162 => add_123, add_124, clone_168, mul_123, mul_124, rsqrt_26, sub_71, var_mean_26
# Graph fragment:
#   %add_117 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_126, %arg3_1), kwargs = {})
#   %add_122 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_117, %view_311), kwargs = {})
#   %clone_168 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_122,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_168, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_168, %getitem_59), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_26 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_123,), kwargs = {})
#   %mul_123 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %rsqrt_26), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_123, %arg14_1), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_124, %arg15_1), kwargs = {})
triton_red_fused_add_native_layer_norm_14 = async_compile.triton('triton_red_fused_add_native_layer_norm_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (150528*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
        tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp13 - tmp10
        tmp15 = 768.0
        tmp16 = tmp11 / tmp15
        tmp17 = 1e-06
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qk/cqkk4zoj2kdqa2acoz7ocrtpm77ksse3ngmrezexrrbzlqlxftoe.py
# Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_164 => add_125, erf_12, mul_125, mul_126, mul_127
# Graph fragment:
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_313, 0.5), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_313, 0.7071067811865476), kwargs = {})
#   %erf_12 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_126,), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_12, 1), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_125, %add_125), kwargs = {})
triton_poi_fused_gelu_15 = async_compile.triton('triton_poi_fused_gelu_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
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


# kernel path: /tmp/torchinductor_sahanp/am/camrkot36uvrei3vntdjqih3j3tscaclerffmgly6qoqycajljis.py
# Topologically Sorted Source Nodes: [x_168, x_169], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_168 => add_126
#   x_169 => add_127, add_128, clone_171, mul_128, mul_129, rsqrt_27, sub_72, var_mean_27
# Graph fragment:
#   %add_126 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_122, %view_315), kwargs = {})
#   %clone_171 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_126,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_171, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_171, %getitem_61), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_60, 1e-06), kwargs = {})
#   %rsqrt_27 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_127,), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %rsqrt_27), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_128, %arg20_1), kwargs = {})
#   %add_128 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_129, %arg21_1), kwargs = {})
triton_per_fused_add_native_layer_norm_16 = async_compile.triton('triton_per_fused_add_native_layer_norm_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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


# kernel path: /tmp/torchinductor_sahanp/ts/ctsvpa7ulxoe53qw4rhyu5bft2h52qulunygtun32n5i3llqs7rt.py
# Topologically Sorted Source Nodes: [x_168, x_173, x_174], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_168 => add_126
#   x_173 => add_131
#   x_174 => add_132, add_133, clone_180, mul_133, mul_134, rsqrt_28, sub_76, var_mean_28
# Graph fragment:
#   %add_126 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_122, %view_315), kwargs = {})
#   %add_131 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_126, %view_333), kwargs = {})
#   %clone_180 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_131,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_28 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_180, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_76 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_180, %getitem_63), kwargs = {})
#   %add_132 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_62, 1e-06), kwargs = {})
#   %rsqrt_28 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_132,), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_76, %rsqrt_28), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %arg29_1), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, %arg30_1), kwargs = {})
triton_per_fused_add_native_layer_norm_17 = async_compile.triton('triton_per_fused_add_native_layer_norm_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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


# kernel path: /tmp/torchinductor_sahanp/4n/c4nx7v2u5zekpba4x4a2rvzrjuvdstizjwwrzw3hphczzzqhkeap.py
# Topologically Sorted Source Nodes: [x_277, x_278], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_277 => cat_1
#   x_278 => add_208, add_209, mul_218, mul_219, rsqrt_45, sub_117, var_mean_45
# Graph fragment:
#   %cat_1 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_79, %add_207], 1), kwargs = {})
#   %var_mean_45 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_117 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_1, %getitem_97), kwargs = {})
#   %add_208 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_96, 1e-06), kwargs = {})
#   %rsqrt_45 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_208,), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_117, %rsqrt_45), kwargs = {})
#   %mul_219 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_218, %arg155_1), kwargs = {})
#   %add_209 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_219, %arg156_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_18 = async_compile.triton('triton_per_fused_cat_native_layer_norm_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, xnumel, rnumel):
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
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp40 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (768*((-1) + x0)) + (150528*x1)), rmask & tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r2 + (768*((-1) + x0)) + (150528*x1)), rmask & tmp6, other=0.0)
    tmp11 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
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
    tmp24 = tl.full([1], 768, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 768.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp43, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/td/ctdonhdblu6tvigju542vucjpsbt33eqfc7yuajxotxntegrgwzx.py
# Topologically Sorted Source Nodes: [matmul_44], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_44 => clone_279
# Graph fragment:
#   %clone_279 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_130,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_19 = async_compile.triton('triton_poi_fused_clone_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 197
    x2 = (xindex // 9456) % 16
    x3 = (xindex // 151296)
    x4 = xindex % 9456
    x5 = (xindex // 9456)
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (2304*x1) + (453888*x3)), xmask)
    tl.store(out_ptr0 + (x4 + (9472*x5)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/no/cnofxecat4editbopa5fwejpp7hpyubu4bf22i5z6dya4ainxll2.py
# Topologically Sorted Source Nodes: [matmul_44], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_44 => clone_280
# Graph fragment:
#   %clone_280 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_131,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_20 = async_compile.triton('triton_poi_fused_clone_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 197
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = (yindex // 768)
    y4 = yindex % 768
    y0 = yindex % 48
    y5 = (yindex // 48)
    tmp0 = tl.load(in_ptr0 + (768 + y4 + (2304*x3) + (453888*y2)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (197*y0) + (9472*y5)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/u2/cu2lp5uit4j2esqnoqo4jvwmzwuz2iw4aigi33i24guoc5zl746c.py
# Topologically Sorted Source Nodes: [attn_67], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_67 => div_62, exp_42, sum_63
# Graph fragment:
#   %mul_tensor_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_519, 1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_2, [-1], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_2, %amax_default_1), kwargs = {})
#   %mul_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_1, 0.14433756729740643), kwargs = {})
#   %exp_42 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_3,), kwargs = {})
#   %sum_63 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_42, [-1], True), kwargs = {})
#   %div_62 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_42, %sum_63), kwargs = {})
triton_red_fused__softmax_21 = async_compile.triton('triton_red_fused__softmax_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_21(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25216
    rnumel = 197
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x5 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x0 = xindex % 3152
    x1 = (xindex // 3152)
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
        tmp10 = 0.14433756729740643
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
        tmp20 = 0.14433756729740643
        tmp21 = tmp19 * tmp20
        tmp22 = tl_math.exp(tmp21)
        tmp23 = tmp22 / tmp14
        tl.store(out_ptr2 + (r2 + (197*x3) + (38816*x6)), tmp23, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/z6/cz6nudd3bq2opo25my7vlejty57pzbawl5f42ksxrx5xq5c5ii4u.py
# Topologically Sorted Source Nodes: [matmul_45], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_45 => clone_282
# Graph fragment:
#   %clone_282 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_133,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_22 = async_compile.triton('triton_poi_fused_clone_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 197
    x2 = (xindex // 9456) % 16
    x3 = (xindex // 151296)
    x4 = xindex % 9456
    x5 = (xindex // 9456)
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + (48*x2) + (2304*x1) + (453888*x3)), xmask)
    tl.store(out_ptr0 + (x4 + (9472*x5)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/q5/cq5mrnu6jifeggwtknwdjkvwckksanam632pacg47yipsujlob7s.py
# Topologically Sorted Source Nodes: [x_279], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_279 => clone_283
# Graph fragment:
#   %clone_283 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_240,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_23 = async_compile.triton('triton_poi_fused_clone_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 16
    x2 = (xindex // 768) % 197
    x3 = (xindex // 151296)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (9456*x1) + (151296*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bc/cbcb4lezffpmclc6bvi6kp7uhy223fvuvfhydboiy6l4zmshwvs6.py
# Topologically Sorted Source Nodes: [x_277, x_282, x_283], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_277 => cat_1
#   x_282 => add_210
#   x_283 => add_211, add_212, mul_221, mul_222, rsqrt_46, sub_119, var_mean_46
# Graph fragment:
#   %cat_1 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_79, %add_207], 1), kwargs = {})
#   %add_210 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_1, %view_525), kwargs = {})
#   %var_mean_46 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_210, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_119 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_210, %getitem_102), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_101, 1e-06), kwargs = {})
#   %rsqrt_46 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_211,), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_119, %rsqrt_46), kwargs = {})
#   %mul_222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_221, %arg160_1), kwargs = {})
#   %add_212 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_222, %arg161_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_24 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
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
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (768*((-1) + x0)) + (150528*x1)), rmask & tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r2 + (768*((-1) + x0)) + (150528*x1)), rmask & tmp6, other=0.0)
    tmp11 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
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
    tmp28 = tl.full([1], 768, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = tmp20 - tmp30
    tmp38 = 768.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp20, rmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp47, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ed/cedsgxg6y4qnqfcvyoaw7jcj6zwiwqxtvrt64h3wiev2ox5t6bdd.py
# Topologically Sorted Source Nodes: [x_285], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_285 => add_213, erf_22, mul_223, mul_224, mul_225
# Graph fragment:
#   %mul_223 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_527, 0.5), kwargs = {})
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_527, 0.7071067811865476), kwargs = {})
#   %erf_22 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_224,), kwargs = {})
#   %add_213 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_22, 1), kwargs = {})
#   %mul_225 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_223, %add_213), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/om/com7pwaw5gmltyw73ibu54y6tm2i7fbqdifuozae5fwaqrsvgpbi.py
# Topologically Sorted Source Nodes: [x_289, x_290], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_289 => add_214
#   x_290 => add_215, add_216, mul_226, mul_227, rsqrt_47, sub_120, var_mean_47
# Graph fragment:
#   %add_214 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_210, %view_529), kwargs = {})
#   %var_mean_47 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_214, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_214, %getitem_104), kwargs = {})
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_103, 1e-06), kwargs = {})
#   %rsqrt_47 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_215,), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_120, %rsqrt_47), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_226, %arg166_1), kwargs = {})
#   %add_216 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_227, %arg167_1), kwargs = {})
triton_per_fused_add_native_layer_norm_26 = async_compile.triton('triton_per_fused_add_native_layer_norm_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_sahanp/4u/c4u2vxjoyn4wlvai6usbgwdy3kkhvn44dqj4xxevq4wj5vh655v3.py
# Topologically Sorted Source Nodes: [x_289, x_294, x_295], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_289 => add_214
#   x_294 => add_217
#   x_295 => add_218, add_219, mul_229, mul_230, rsqrt_48, sub_122, var_mean_48
# Graph fragment:
#   %add_214 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_210, %view_529), kwargs = {})
#   %add_217 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_214, %view_541), kwargs = {})
#   %var_mean_48 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_217, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_122 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_217, %getitem_109), kwargs = {})
#   %add_218 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_108, 1e-06), kwargs = {})
#   %rsqrt_48 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_218,), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_122, %rsqrt_48), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_229, %arg171_1), kwargs = {})
#   %add_219 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_230, %arg172_1), kwargs = {})
triton_per_fused_add_native_layer_norm_27 = async_compile.triton('triton_per_fused_add_native_layer_norm_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_sahanp/bp/cbpbdv7gr46tt7xx5pjvg67tsdrzbr5tdjemztz2nso5bse532kp.py
# Topologically Sorted Source Nodes: [x_301, x_302], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_301 => add_221
#   x_302 => var_mean_49
# Graph fragment:
#   %add_221 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_217, %view_545), kwargs = {})
#   %var_mean_49 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_221, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_28 = async_compile.triton('triton_per_fused_add_native_layer_norm_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_28(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_sahanp/f7/cf72rqpomwgkazzverp2xfu2j5jeeyd7wih47ss6ga5pnccxfmru.py
# Topologically Sorted Source Nodes: [x_304], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_304 => clone_295
# Graph fragment:
#   %clone_295 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%select_121,), kwargs = {})
triton_poi_fused_clone_29 = async_compile.triton('triton_poi_fused_clone_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (1, 196, 768), (150528, 768, 1))
    assert_size_stride(arg4_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (1536, 768), (768, 1))
    assert_size_stride(arg8_1, (16, 3), (3, 1))
    assert_size_stride(arg9_1, (16, ), (1, ))
    assert_size_stride(arg10_1, (16, ), (1, ))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (768, 768), (768, 1))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (3072, 768), (768, 1))
    assert_size_stride(arg17_1, (3072, ), (1, ))
    assert_size_stride(arg18_1, (768, 3072), (3072, 1))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (1536, 768), (768, 1))
    assert_size_stride(arg23_1, (16, 3), (3, 1))
    assert_size_stride(arg24_1, (16, ), (1, ))
    assert_size_stride(arg25_1, (16, ), (1, ))
    assert_size_stride(arg26_1, (768, 768), (768, 1))
    assert_size_stride(arg27_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (3072, 768), (768, 1))
    assert_size_stride(arg32_1, (3072, ), (1, ))
    assert_size_stride(arg33_1, (768, 3072), (3072, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (1536, 768), (768, 1))
    assert_size_stride(arg38_1, (16, 3), (3, 1))
    assert_size_stride(arg39_1, (16, ), (1, ))
    assert_size_stride(arg40_1, (16, ), (1, ))
    assert_size_stride(arg41_1, (768, 768), (768, 1))
    assert_size_stride(arg42_1, (768, 768), (768, 1))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (3072, 768), (768, 1))
    assert_size_stride(arg47_1, (3072, ), (1, ))
    assert_size_stride(arg48_1, (768, 3072), (3072, 1))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (1536, 768), (768, 1))
    assert_size_stride(arg53_1, (16, 3), (3, 1))
    assert_size_stride(arg54_1, (16, ), (1, ))
    assert_size_stride(arg55_1, (16, ), (1, ))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
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
    assert_size_stride(arg67_1, (1536, 768), (768, 1))
    assert_size_stride(arg68_1, (16, 3), (3, 1))
    assert_size_stride(arg69_1, (16, ), (1, ))
    assert_size_stride(arg70_1, (16, ), (1, ))
    assert_size_stride(arg71_1, (768, 768), (768, 1))
    assert_size_stride(arg72_1, (768, 768), (768, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (3072, 768), (768, 1))
    assert_size_stride(arg77_1, (3072, ), (1, ))
    assert_size_stride(arg78_1, (768, 3072), (3072, 1))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (1536, 768), (768, 1))
    assert_size_stride(arg83_1, (16, 3), (3, 1))
    assert_size_stride(arg84_1, (16, ), (1, ))
    assert_size_stride(arg85_1, (16, ), (1, ))
    assert_size_stride(arg86_1, (768, 768), (768, 1))
    assert_size_stride(arg87_1, (768, 768), (768, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (3072, 768), (768, 1))
    assert_size_stride(arg92_1, (3072, ), (1, ))
    assert_size_stride(arg93_1, (768, 3072), (3072, 1))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (768, ), (1, ))
    assert_size_stride(arg97_1, (1536, 768), (768, 1))
    assert_size_stride(arg98_1, (16, 3), (3, 1))
    assert_size_stride(arg99_1, (16, ), (1, ))
    assert_size_stride(arg100_1, (16, ), (1, ))
    assert_size_stride(arg101_1, (768, 768), (768, 1))
    assert_size_stride(arg102_1, (768, 768), (768, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (3072, 768), (768, 1))
    assert_size_stride(arg107_1, (3072, ), (1, ))
    assert_size_stride(arg108_1, (768, 3072), (3072, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (1536, 768), (768, 1))
    assert_size_stride(arg113_1, (16, 3), (3, 1))
    assert_size_stride(arg114_1, (16, ), (1, ))
    assert_size_stride(arg115_1, (16, ), (1, ))
    assert_size_stride(arg116_1, (768, 768), (768, 1))
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
    assert_size_stride(arg127_1, (1536, 768), (768, 1))
    assert_size_stride(arg128_1, (16, 3), (3, 1))
    assert_size_stride(arg129_1, (16, ), (1, ))
    assert_size_stride(arg130_1, (16, ), (1, ))
    assert_size_stride(arg131_1, (768, 768), (768, 1))
    assert_size_stride(arg132_1, (768, 768), (768, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (3072, 768), (768, 1))
    assert_size_stride(arg137_1, (3072, ), (1, ))
    assert_size_stride(arg138_1, (768, 3072), (3072, 1))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (1536, 768), (768, 1))
    assert_size_stride(arg143_1, (16, 3), (3, 1))
    assert_size_stride(arg144_1, (16, ), (1, ))
    assert_size_stride(arg145_1, (16, ), (1, ))
    assert_size_stride(arg146_1, (768, 768), (768, 1))
    assert_size_stride(arg147_1, (768, 768), (768, 1))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (3072, 768), (768, 1))
    assert_size_stride(arg152_1, (3072, ), (1, ))
    assert_size_stride(arg153_1, (768, 3072), (3072, 1))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (2304, 768), (768, 1))
    assert_size_stride(arg158_1, (768, 768), (768, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (3072, 768), (768, 1))
    assert_size_stride(arg163_1, (3072, ), (1, ))
    assert_size_stride(arg164_1, (768, 3072), (3072, 1))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (2304, 768), (768, 1))
    assert_size_stride(arg169_1, (768, 768), (768, 1))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (3072, 768), (768, 1))
    assert_size_stride(arg174_1, (3072, ), (1, ))
    assert_size_stride(arg175_1, (768, 3072), (3072, 1))
    assert_size_stride(arg176_1, (768, ), (1, ))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (1000, 768), (768, 1))
    assert_size_stride(arg180_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_153], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((8, 196, 1, 6), (1184, 6, 9472, 1), torch.float32)
        buf2 = empty_strided_cuda((8, 196, 1, 6), (1184, 6, 9472, 1), torch.float32)
        buf3 = empty_strided_cuda((8, 196, 1, 6), (1184, 6, 9472, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_155, x_157], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_0.run(buf0, arg2_1, arg3_1, buf1, buf2, buf3, 9408, 128, grid=grid(9408), stream=stream0)
        buf4 = empty_strided_cuda((8, 196, 1), (196, 1, 1568), torch.float32)
        buf5 = empty_strided_cuda((8, 196, 1), (196, 1, 1568), torch.float32)
        # Topologically Sorted Source Nodes: [x_155, x_157], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 1568, 6, grid=grid(1568), stream=stream0)
        del buf1
        del buf2
        del buf3
        buf7 = empty_strided_cuda((8, 196, 768), (150528, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_155, x_157], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_2.run(buf0, arg2_1, arg3_1, buf4, buf5, arg5_1, arg6_1, buf7, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg5_1
        del arg6_1
        del buf4
        del buf5
        buf8 = empty_strided_cuda((1568, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_69], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (1568, 768), (768, 1), 0), reinterpret_tensor(arg7_1, (768, 1536), (1, 768), 0), out=buf8)
        del arg7_1
        buf9 = empty_strided_cuda((8, 16, 196, 48), (150528, 9408, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf8, buf9, 1204224, grid=grid(1204224), stream=stream0)
        buf10 = empty_strided_cuda((8, 16, 48, 196), (150528, 9408, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf8, buf10, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf11 = empty_strided_cuda((128, 196, 196), (38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf10, (128, 48, 196), (9408, 196, 1), 0), out=buf11)
        buf12 = empty_strided_cuda((8, 16, 196, 1), (3136, 196, 1, 25088), torch.float32)
        buf13 = empty_strided_cuda((8, 16, 196, 1), (3136, 196, 1, 25088), torch.float32)
        # Topologically Sorted Source Nodes: [patch_score_21], Original ATen: [aten._softmax]
        triton_per_fused__softmax_5.run(buf11, buf12, buf13, 25088, 196, grid=grid(25088), stream=stream0)
        buf14 = empty_strided_cuda((1, 196, 196, 3), (115248, 588, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rel_indices, setitem, setitem_1, setitem_2, to], Original ATen: [aten.zeros, aten.copy, aten._to_copy]
        triton_poi_fused__to_copy_copy_zeros_6.run(buf14, 115248, grid=grid(115248), stream=stream0)
        buf15 = empty_strided_cuda((8, 196, 196, 3), (115264, 588, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf14, buf15, 921984, grid=grid(921984), stream=stream0)
        buf16 = empty_strided_cuda((307328, 3), (3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.mm]
        triton_poi_fused_mm_8.run(buf15, buf16, 921984, grid=grid(921984), stream=stream0)
        buf17 = empty_strided_cuda((307328, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.mm]
        extern_kernels.mm(buf16, reinterpret_tensor(arg8_1, (3, 16), (1, 3), 0), out=buf17)
        del arg8_1
        buf18 = empty_strided_cuda((8, 16, 196, 1), (3136, 1, 16, 25088), torch.float32)
        buf19 = empty_strided_cuda((8, 16, 196, 1), (3136, 1, 16, 25088), torch.float32)
        # Topologically Sorted Source Nodes: [pos_score_32], Original ATen: [aten._softmax]
        triton_red_fused__softmax_9.run(buf17, arg9_1, buf18, buf19, 25088, 196, grid=grid(25088), stream=stream0)
        buf20 = empty_strided_cuda((8, 16, 196, 196), (614912, 38432, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_20, sub_20, patch_score_21, mul_33, sigmoid_21, pos_score_32, mul_34, attn_36], Original ATen: [aten.sigmoid, aten.rsub, aten._softmax, aten.mul, aten.add]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_10.run(arg10_1, buf11, buf12, buf13, buf17, arg9_1, buf18, buf19, buf20, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg10_1
        del arg9_1
        buf23 = empty_strided_cuda((8, 16, 196, 196), (614912, 38432, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sum_11, attn_37], Original ATen: [aten.sum, aten.div]
        triton_red_fused_div_sum_11.run(buf20, buf23, 25088, 196, grid=grid(25088), stream=stream0)
        buf22 = reinterpret_tensor(buf9, (1568, 768), (768, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (1568, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), out=buf22)
        del arg11_1
        buf24 = reinterpret_tensor(buf7, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf22, buf24, 1204224, grid=grid(1204224), stream=stream0)
        buf25 = reinterpret_tensor(buf22, (128, 196, 48), (9408, 48, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf23, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf24, (128, 196, 48), (9408, 48, 1), 0), out=buf25)
        buf26 = reinterpret_tensor(buf24, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf25, buf26, 1204224, grid=grid(1204224), stream=stream0)
        buf27 = reinterpret_tensor(buf25, (1568, 768), (768, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (1568, 768), (768, 1), 0), reinterpret_tensor(arg12_1, (768, 768), (1, 768), 0), out=buf27)
        del arg12_1
        buf28 = reinterpret_tensor(buf27, (8, 196, 768), (150528, 768, 1), 0); del buf27  # reuse
        buf32 = reinterpret_tensor(buf26, (8, 196, 768), (150528, 768, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_155, x_161, x_162], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_14.run(buf28, buf0, arg2_1, arg3_1, arg13_1, arg14_1, arg15_1, buf32, 1568, 768, grid=grid(1568), stream=stream0)
        del arg13_1
        del arg14_1
        del arg15_1
        del arg2_1
        del arg3_1
        buf33 = empty_strided_cuda((1568, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (1568, 768), (768, 1), 0), reinterpret_tensor(arg16_1, (768, 3072), (1, 768), 0), out=buf33)
        del arg16_1
        buf34 = reinterpret_tensor(buf33, (8, 196, 3072), (602112, 3072, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf34, arg17_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg17_1
        buf35 = reinterpret_tensor(buf32, (1568, 768), (768, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg18_1, (3072, 768), (1, 3072), 0), out=buf35)
        del arg18_1
        buf39 = reinterpret_tensor(buf0, (8, 196, 768), (150528, 768, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_168, x_169], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf28, buf35, arg19_1, arg20_1, arg21_1, buf39, 1568, 768, grid=grid(1568), stream=stream0)
        del arg20_1
        del arg21_1
        buf40 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [linear_75], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (1568, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 1536), (1, 768), 0), out=buf40)
        del arg22_1
        buf41 = reinterpret_tensor(buf10, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf40, buf41, 1204224, grid=grid(1204224), stream=stream0)
        buf42 = empty_strided_cuda((8, 16, 48, 196), (150528, 9408, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf40, buf42, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf43 = reinterpret_tensor(buf17, (128, 196, 196), (38416, 196, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf41, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf42, (128, 48, 196), (9408, 196, 1), 0), out=buf43)
        buf44 = reinterpret_tensor(buf19, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf19  # reuse
        buf45 = reinterpret_tensor(buf18, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [patch_score_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_5.run(buf43, buf44, buf45, 25088, 196, grid=grid(25088), stream=stream0)
        buf46 = empty_strided_cuda((1, 196, 196, 3), (115248, 588, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rel_indices_1, setitem_3, setitem_4, setitem_5, to_1], Original ATen: [aten.zeros, aten.copy, aten._to_copy]
        triton_poi_fused__to_copy_copy_zeros_6.run(buf46, 115248, grid=grid(115248), stream=stream0)
        buf47 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [linear_76], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf46, buf47, 921984, grid=grid(921984), stream=stream0)
        buf48 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [linear_76], Original ATen: [aten.mm]
        triton_poi_fused_mm_8.run(buf47, buf48, 921984, grid=grid(921984), stream=stream0)
        buf49 = reinterpret_tensor(buf11, (307328, 16), (16, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [linear_76], Original ATen: [aten.mm]
        extern_kernels.mm(buf48, reinterpret_tensor(arg23_1, (3, 16), (1, 3), 0), out=buf49)
        del arg23_1
        buf50 = reinterpret_tensor(buf13, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf13  # reuse
        buf51 = reinterpret_tensor(buf12, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [pos_score_35], Original ATen: [aten._softmax]
        triton_red_fused__softmax_9.run(buf49, arg24_1, buf50, buf51, 25088, 196, grid=grid(25088), stream=stream0)
        buf52 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_22, sub_21, patch_score_23, mul_36, sigmoid_23, pos_score_35, mul_37, attn_39], Original ATen: [aten.sigmoid, aten.rsub, aten._softmax, aten.mul, aten.add]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_10.run(arg25_1, buf43, buf44, buf45, buf49, arg24_1, buf50, buf51, buf52, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg24_1
        del arg25_1
        buf55 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [sum_12, attn_40], Original ATen: [aten.sum, aten.div]
        triton_red_fused_div_sum_11.run(buf52, buf55, 25088, 196, grid=grid(25088), stream=stream0)
        buf54 = reinterpret_tensor(buf42, (1568, 768), (768, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (1568, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 768), (1, 768), 0), out=buf54)
        del arg26_1
        buf56 = reinterpret_tensor(buf39, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf54, buf56, 1204224, grid=grid(1204224), stream=stream0)
        buf57 = reinterpret_tensor(buf54, (128, 196, 48), (9408, 48, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf55, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf56, (128, 196, 48), (9408, 48, 1), 0), out=buf57)
        buf58 = reinterpret_tensor(buf56, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_170], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf57, buf58, 1204224, grid=grid(1204224), stream=stream0)
        buf59 = reinterpret_tensor(buf57, (1568, 768), (768, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (1568, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), out=buf59)
        del arg27_1
        buf60 = reinterpret_tensor(buf59, (8, 196, 768), (150528, 768, 1), 0); del buf59  # reuse
        buf64 = reinterpret_tensor(buf58, (8, 196, 768), (150528, 768, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_168, x_173, x_174], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf60, buf28, buf35, arg19_1, arg28_1, arg29_1, arg30_1, buf64, 1568, 768, grid=grid(1568), stream=stream0)
        del arg19_1
        del arg28_1
        del arg29_1
        del arg30_1
        buf65 = reinterpret_tensor(buf34, (1568, 3072), (3072, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (1568, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 3072), (1, 768), 0), out=buf65)
        del arg31_1
        buf66 = reinterpret_tensor(buf65, (8, 196, 3072), (602112, 3072, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_176], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf66, arg32_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg32_1
        buf67 = reinterpret_tensor(buf64, (1568, 768), (768, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf66, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg33_1, (3072, 768), (1, 3072), 0), out=buf67)
        del arg33_1
        buf71 = reinterpret_tensor(buf35, (8, 196, 768), (150528, 768, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [x_180, x_181], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf60, buf67, arg34_1, arg35_1, arg36_1, buf71, 1568, 768, grid=grid(1568), stream=stream0)
        del arg35_1
        del arg36_1
        buf72 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [linear_81], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf71, (1568, 768), (768, 1), 0), reinterpret_tensor(arg37_1, (768, 1536), (1, 768), 0), out=buf72)
        del arg37_1
        buf73 = reinterpret_tensor(buf28, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf72, buf73, 1204224, grid=grid(1204224), stream=stream0)
        buf74 = reinterpret_tensor(buf41, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf72, buf74, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf75 = reinterpret_tensor(buf49, (128, 196, 196), (38416, 196, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf74, (128, 48, 196), (9408, 196, 1), 0), out=buf75)
        buf76 = reinterpret_tensor(buf51, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf51  # reuse
        buf77 = reinterpret_tensor(buf50, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [patch_score_25], Original ATen: [aten._softmax]
        triton_per_fused__softmax_5.run(buf75, buf76, buf77, 25088, 196, grid=grid(25088), stream=stream0)
        buf78 = empty_strided_cuda((1, 196, 196, 3), (115248, 588, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rel_indices_2, setitem_6, setitem_7, setitem_8, to_2], Original ATen: [aten.zeros, aten.copy, aten._to_copy]
        triton_poi_fused__to_copy_copy_zeros_6.run(buf78, 115248, grid=grid(115248), stream=stream0)
        buf79 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf78, buf79, 921984, grid=grid(921984), stream=stream0)
        buf80 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.mm]
        triton_poi_fused_mm_8.run(buf79, buf80, 921984, grid=grid(921984), stream=stream0)
        buf81 = reinterpret_tensor(buf43, (307328, 16), (16, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.mm]
        extern_kernels.mm(buf80, reinterpret_tensor(arg38_1, (3, 16), (1, 3), 0), out=buf81)
        del arg38_1
        buf82 = reinterpret_tensor(buf45, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf45  # reuse
        buf83 = reinterpret_tensor(buf44, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [pos_score_38], Original ATen: [aten._softmax]
        triton_red_fused__softmax_9.run(buf81, arg39_1, buf82, buf83, 25088, 196, grid=grid(25088), stream=stream0)
        buf84 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_24, sub_22, patch_score_25, mul_39, sigmoid_25, pos_score_38, mul_40, attn_42], Original ATen: [aten.sigmoid, aten.rsub, aten._softmax, aten.mul, aten.add]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_10.run(arg40_1, buf75, buf76, buf77, buf81, arg39_1, buf82, buf83, buf84, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg39_1
        del arg40_1
        buf87 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [sum_13, attn_43], Original ATen: [aten.sum, aten.div]
        triton_red_fused_div_sum_11.run(buf84, buf87, 25088, 196, grid=grid(25088), stream=stream0)
        buf86 = reinterpret_tensor(buf74, (1568, 768), (768, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [linear_83], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf71, (1568, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), out=buf86)
        del arg41_1
        buf88 = reinterpret_tensor(buf71, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf86, buf88, 1204224, grid=grid(1204224), stream=stream0)
        buf89 = reinterpret_tensor(buf86, (128, 196, 48), (9408, 48, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf87, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf88, (128, 196, 48), (9408, 48, 1), 0), out=buf89)
        buf90 = reinterpret_tensor(buf88, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_182], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf89, buf90, 1204224, grid=grid(1204224), stream=stream0)
        buf91 = reinterpret_tensor(buf89, (1568, 768), (768, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (1568, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 768), (1, 768), 0), out=buf91)
        del arg42_1
        buf92 = reinterpret_tensor(buf91, (8, 196, 768), (150528, 768, 1), 0); del buf91  # reuse
        buf96 = reinterpret_tensor(buf90, (8, 196, 768), (150528, 768, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_180, x_185, x_186], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf92, buf60, buf67, arg34_1, arg43_1, arg44_1, arg45_1, buf96, 1568, 768, grid=grid(1568), stream=stream0)
        del arg34_1
        del arg43_1
        del arg44_1
        del arg45_1
        buf97 = reinterpret_tensor(buf66, (1568, 3072), (3072, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (1568, 768), (768, 1), 0), reinterpret_tensor(arg46_1, (768, 3072), (1, 768), 0), out=buf97)
        del arg46_1
        buf98 = reinterpret_tensor(buf97, (8, 196, 3072), (602112, 3072, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf98, arg47_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg47_1
        buf99 = reinterpret_tensor(buf96, (1568, 768), (768, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg48_1, (3072, 768), (1, 3072), 0), out=buf99)
        del arg48_1
        buf103 = reinterpret_tensor(buf67, (8, 196, 768), (150528, 768, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_192, x_193], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf92, buf99, arg49_1, arg50_1, arg51_1, buf103, 1568, 768, grid=grid(1568), stream=stream0)
        del arg50_1
        del arg51_1
        buf104 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [linear_87], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (1568, 768), (768, 1), 0), reinterpret_tensor(arg52_1, (768, 1536), (1, 768), 0), out=buf104)
        del arg52_1
        buf105 = reinterpret_tensor(buf60, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf104, buf105, 1204224, grid=grid(1204224), stream=stream0)
        buf106 = reinterpret_tensor(buf73, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf104, buf106, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf107 = reinterpret_tensor(buf81, (128, 196, 196), (38416, 196, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf105, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf106, (128, 48, 196), (9408, 196, 1), 0), out=buf107)
        buf108 = reinterpret_tensor(buf83, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf83  # reuse
        buf109 = reinterpret_tensor(buf82, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [patch_score_27], Original ATen: [aten._softmax]
        triton_per_fused__softmax_5.run(buf107, buf108, buf109, 25088, 196, grid=grid(25088), stream=stream0)
        buf110 = empty_strided_cuda((1, 196, 196, 3), (115248, 588, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rel_indices_3, setitem_9, setitem_10, setitem_11, to_3], Original ATen: [aten.zeros, aten.copy, aten._to_copy]
        triton_poi_fused__to_copy_copy_zeros_6.run(buf110, 115248, grid=grid(115248), stream=stream0)
        buf111 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf110, buf111, 921984, grid=grid(921984), stream=stream0)
        buf112 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.mm]
        triton_poi_fused_mm_8.run(buf111, buf112, 921984, grid=grid(921984), stream=stream0)
        buf113 = reinterpret_tensor(buf75, (307328, 16), (16, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.mm]
        extern_kernels.mm(buf112, reinterpret_tensor(arg53_1, (3, 16), (1, 3), 0), out=buf113)
        del arg53_1
        buf114 = reinterpret_tensor(buf77, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf77  # reuse
        buf115 = reinterpret_tensor(buf76, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [pos_score_41], Original ATen: [aten._softmax]
        triton_red_fused__softmax_9.run(buf113, arg54_1, buf114, buf115, 25088, 196, grid=grid(25088), stream=stream0)
        buf116 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_26, sub_23, patch_score_27, mul_42, sigmoid_27, pos_score_41, mul_43, attn_45], Original ATen: [aten.sigmoid, aten.rsub, aten._softmax, aten.mul, aten.add]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_10.run(arg55_1, buf107, buf108, buf109, buf113, arg54_1, buf114, buf115, buf116, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg54_1
        del arg55_1
        buf119 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [sum_14, attn_46], Original ATen: [aten.sum, aten.div]
        triton_red_fused_div_sum_11.run(buf116, buf119, 25088, 196, grid=grid(25088), stream=stream0)
        buf118 = reinterpret_tensor(buf106, (1568, 768), (768, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (1568, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), out=buf118)
        del arg56_1
        buf120 = reinterpret_tensor(buf103, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [matmul_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf118, buf120, 1204224, grid=grid(1204224), stream=stream0)
        buf121 = reinterpret_tensor(buf118, (128, 196, 48), (9408, 48, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf120, (128, 196, 48), (9408, 48, 1), 0), out=buf121)
        buf122 = reinterpret_tensor(buf120, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [x_194], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf121, buf122, 1204224, grid=grid(1204224), stream=stream0)
        buf123 = reinterpret_tensor(buf121, (1568, 768), (768, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (1568, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), out=buf123)
        del arg57_1
        buf124 = reinterpret_tensor(buf123, (8, 196, 768), (150528, 768, 1), 0); del buf123  # reuse
        buf128 = reinterpret_tensor(buf122, (8, 196, 768), (150528, 768, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_192, x_197, x_198], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf124, buf92, buf99, arg49_1, arg58_1, arg59_1, arg60_1, buf128, 1568, 768, grid=grid(1568), stream=stream0)
        del arg49_1
        del arg58_1
        del arg59_1
        del arg60_1
        buf129 = reinterpret_tensor(buf98, (1568, 3072), (3072, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (1568, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 3072), (1, 768), 0), out=buf129)
        del arg61_1
        buf130 = reinterpret_tensor(buf129, (8, 196, 3072), (602112, 3072, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [x_200], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf130, arg62_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg62_1
        buf131 = reinterpret_tensor(buf128, (1568, 768), (768, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg63_1, (3072, 768), (1, 3072), 0), out=buf131)
        del arg63_1
        buf135 = reinterpret_tensor(buf99, (8, 196, 768), (150528, 768, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [x_204, x_205], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf124, buf131, arg64_1, arg65_1, arg66_1, buf135, 1568, 768, grid=grid(1568), stream=stream0)
        del arg65_1
        del arg66_1
        buf136 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (1568, 768), (768, 1), 0), reinterpret_tensor(arg67_1, (768, 1536), (1, 768), 0), out=buf136)
        del arg67_1
        buf137 = reinterpret_tensor(buf92, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [matmul_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf136, buf137, 1204224, grid=grid(1204224), stream=stream0)
        buf138 = reinterpret_tensor(buf105, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [matmul_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf136, buf138, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf139 = reinterpret_tensor(buf113, (128, 196, 196), (38416, 196, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [matmul_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf138, (128, 48, 196), (9408, 196, 1), 0), out=buf139)
        buf140 = reinterpret_tensor(buf115, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf115  # reuse
        buf141 = reinterpret_tensor(buf114, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [patch_score_29], Original ATen: [aten._softmax]
        triton_per_fused__softmax_5.run(buf139, buf140, buf141, 25088, 196, grid=grid(25088), stream=stream0)
        buf142 = empty_strided_cuda((1, 196, 196, 3), (115248, 588, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rel_indices_4, setitem_12, setitem_13, setitem_14, to_4], Original ATen: [aten.zeros, aten.copy, aten._to_copy]
        triton_poi_fused__to_copy_copy_zeros_6.run(buf142, 115248, grid=grid(115248), stream=stream0)
        buf143 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [linear_94], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf142, buf143, 921984, grid=grid(921984), stream=stream0)
        buf144 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [linear_94], Original ATen: [aten.mm]
        triton_poi_fused_mm_8.run(buf143, buf144, 921984, grid=grid(921984), stream=stream0)
        buf145 = reinterpret_tensor(buf107, (307328, 16), (16, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [linear_94], Original ATen: [aten.mm]
        extern_kernels.mm(buf144, reinterpret_tensor(arg68_1, (3, 16), (1, 3), 0), out=buf145)
        del arg68_1
        buf146 = reinterpret_tensor(buf109, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf109  # reuse
        buf147 = reinterpret_tensor(buf108, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [pos_score_44], Original ATen: [aten._softmax]
        triton_red_fused__softmax_9.run(buf145, arg69_1, buf146, buf147, 25088, 196, grid=grid(25088), stream=stream0)
        buf148 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_28, sub_24, patch_score_29, mul_45, sigmoid_29, pos_score_44, mul_46, attn_48], Original ATen: [aten.sigmoid, aten.rsub, aten._softmax, aten.mul, aten.add]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_10.run(arg70_1, buf139, buf140, buf141, buf145, arg69_1, buf146, buf147, buf148, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg69_1
        del arg70_1
        buf151 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [sum_15, attn_49], Original ATen: [aten.sum, aten.div]
        triton_red_fused_div_sum_11.run(buf148, buf151, 25088, 196, grid=grid(25088), stream=stream0)
        buf150 = reinterpret_tensor(buf138, (1568, 768), (768, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [linear_95], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (1568, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 768), (1, 768), 0), out=buf150)
        del arg71_1
        buf152 = reinterpret_tensor(buf135, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [matmul_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf150, buf152, 1204224, grid=grid(1204224), stream=stream0)
        buf153 = reinterpret_tensor(buf150, (128, 196, 48), (9408, 48, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf151, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf152, (128, 196, 48), (9408, 48, 1), 0), out=buf153)
        buf154 = reinterpret_tensor(buf152, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [x_206], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf153, buf154, 1204224, grid=grid(1204224), stream=stream0)
        buf155 = reinterpret_tensor(buf153, (1568, 768), (768, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf154, (1568, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 768), (1, 768), 0), out=buf155)
        del arg72_1
        buf156 = reinterpret_tensor(buf155, (8, 196, 768), (150528, 768, 1), 0); del buf155  # reuse
        buf160 = reinterpret_tensor(buf154, (8, 196, 768), (150528, 768, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [x_204, x_209, x_210], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf156, buf124, buf131, arg64_1, arg73_1, arg74_1, arg75_1, buf160, 1568, 768, grid=grid(1568), stream=stream0)
        del arg64_1
        del arg73_1
        del arg74_1
        del arg75_1
        buf161 = reinterpret_tensor(buf130, (1568, 3072), (3072, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf160, (1568, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 3072), (1, 768), 0), out=buf161)
        del arg76_1
        buf162 = reinterpret_tensor(buf161, (8, 196, 3072), (602112, 3072, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_212], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf162, arg77_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg77_1
        buf163 = reinterpret_tensor(buf160, (1568, 768), (768, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf162, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg78_1, (3072, 768), (1, 3072), 0), out=buf163)
        del arg78_1
        buf167 = reinterpret_tensor(buf131, (8, 196, 768), (150528, 768, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_216, x_217], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf156, buf163, arg79_1, arg80_1, arg81_1, buf167, 1568, 768, grid=grid(1568), stream=stream0)
        del arg80_1
        del arg81_1
        buf168 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [linear_99], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (1568, 768), (768, 1), 0), reinterpret_tensor(arg82_1, (768, 1536), (1, 768), 0), out=buf168)
        del arg82_1
        buf169 = reinterpret_tensor(buf124, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [matmul_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf168, buf169, 1204224, grid=grid(1204224), stream=stream0)
        buf170 = reinterpret_tensor(buf137, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [matmul_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf168, buf170, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf171 = reinterpret_tensor(buf145, (128, 196, 196), (38416, 196, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [matmul_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf169, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf170, (128, 48, 196), (9408, 196, 1), 0), out=buf171)
        buf172 = reinterpret_tensor(buf147, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf147  # reuse
        buf173 = reinterpret_tensor(buf146, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [patch_score_31], Original ATen: [aten._softmax]
        triton_per_fused__softmax_5.run(buf171, buf172, buf173, 25088, 196, grid=grid(25088), stream=stream0)
        buf174 = empty_strided_cuda((1, 196, 196, 3), (115248, 588, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rel_indices_5, setitem_15, setitem_16, setitem_17, to_5], Original ATen: [aten.zeros, aten.copy, aten._to_copy]
        triton_poi_fused__to_copy_copy_zeros_6.run(buf174, 115248, grid=grid(115248), stream=stream0)
        buf175 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [linear_100], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf174, buf175, 921984, grid=grid(921984), stream=stream0)
        buf176 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [linear_100], Original ATen: [aten.mm]
        triton_poi_fused_mm_8.run(buf175, buf176, 921984, grid=grid(921984), stream=stream0)
        buf177 = reinterpret_tensor(buf139, (307328, 16), (16, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [linear_100], Original ATen: [aten.mm]
        extern_kernels.mm(buf176, reinterpret_tensor(arg83_1, (3, 16), (1, 3), 0), out=buf177)
        del arg83_1
        buf178 = reinterpret_tensor(buf141, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf141  # reuse
        buf179 = reinterpret_tensor(buf140, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [pos_score_47], Original ATen: [aten._softmax]
        triton_red_fused__softmax_9.run(buf177, arg84_1, buf178, buf179, 25088, 196, grid=grid(25088), stream=stream0)
        buf180 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_30, sub_25, patch_score_31, mul_48, sigmoid_31, pos_score_47, mul_49, attn_51], Original ATen: [aten.sigmoid, aten.rsub, aten._softmax, aten.mul, aten.add]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_10.run(arg85_1, buf171, buf172, buf173, buf177, arg84_1, buf178, buf179, buf180, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg84_1
        del arg85_1
        buf183 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [sum_16, attn_52], Original ATen: [aten.sum, aten.div]
        triton_red_fused_div_sum_11.run(buf180, buf183, 25088, 196, grid=grid(25088), stream=stream0)
        buf182 = reinterpret_tensor(buf170, (1568, 768), (768, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [linear_101], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (1568, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), out=buf182)
        del arg86_1
        buf184 = reinterpret_tensor(buf167, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [matmul_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf182, buf184, 1204224, grid=grid(1204224), stream=stream0)
        buf185 = reinterpret_tensor(buf182, (128, 196, 48), (9408, 48, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf183, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf184, (128, 196, 48), (9408, 48, 1), 0), out=buf185)
        buf186 = reinterpret_tensor(buf184, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [x_218], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf185, buf186, 1204224, grid=grid(1204224), stream=stream0)
        buf187 = reinterpret_tensor(buf185, (1568, 768), (768, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf186, (1568, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 768), (1, 768), 0), out=buf187)
        del arg87_1
        buf188 = reinterpret_tensor(buf187, (8, 196, 768), (150528, 768, 1), 0); del buf187  # reuse
        buf192 = reinterpret_tensor(buf186, (8, 196, 768), (150528, 768, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_216, x_221, x_222], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf188, buf156, buf163, arg79_1, arg88_1, arg89_1, arg90_1, buf192, 1568, 768, grid=grid(1568), stream=stream0)
        del arg79_1
        del arg88_1
        del arg89_1
        del arg90_1
        buf193 = reinterpret_tensor(buf162, (1568, 3072), (3072, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf192, (1568, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 3072), (1, 768), 0), out=buf193)
        del arg91_1
        buf194 = reinterpret_tensor(buf193, (8, 196, 3072), (602112, 3072, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [x_224], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf194, arg92_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg92_1
        buf195 = reinterpret_tensor(buf192, (1568, 768), (768, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg93_1, (3072, 768), (1, 3072), 0), out=buf195)
        del arg93_1
        buf199 = reinterpret_tensor(buf163, (8, 196, 768), (150528, 768, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_228, x_229], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf188, buf195, arg94_1, arg95_1, arg96_1, buf199, 1568, 768, grid=grid(1568), stream=stream0)
        del arg95_1
        del arg96_1
        buf200 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [linear_105], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (1568, 768), (768, 1), 0), reinterpret_tensor(arg97_1, (768, 1536), (1, 768), 0), out=buf200)
        del arg97_1
        buf201 = reinterpret_tensor(buf156, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [matmul_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf200, buf201, 1204224, grid=grid(1204224), stream=stream0)
        buf202 = reinterpret_tensor(buf169, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [matmul_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf200, buf202, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf203 = reinterpret_tensor(buf177, (128, 196, 196), (38416, 196, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [matmul_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf201, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf202, (128, 48, 196), (9408, 196, 1), 0), out=buf203)
        buf204 = reinterpret_tensor(buf179, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf179  # reuse
        buf205 = reinterpret_tensor(buf178, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [patch_score_33], Original ATen: [aten._softmax]
        triton_per_fused__softmax_5.run(buf203, buf204, buf205, 25088, 196, grid=grid(25088), stream=stream0)
        buf206 = empty_strided_cuda((1, 196, 196, 3), (115248, 588, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rel_indices_6, setitem_18, setitem_19, setitem_20, to_6], Original ATen: [aten.zeros, aten.copy, aten._to_copy]
        triton_poi_fused__to_copy_copy_zeros_6.run(buf206, 115248, grid=grid(115248), stream=stream0)
        buf207 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf206, buf207, 921984, grid=grid(921984), stream=stream0)
        buf208 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten.mm]
        triton_poi_fused_mm_8.run(buf207, buf208, 921984, grid=grid(921984), stream=stream0)
        buf209 = reinterpret_tensor(buf171, (307328, 16), (16, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten.mm]
        extern_kernels.mm(buf208, reinterpret_tensor(arg98_1, (3, 16), (1, 3), 0), out=buf209)
        del arg98_1
        buf210 = reinterpret_tensor(buf173, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf173  # reuse
        buf211 = reinterpret_tensor(buf172, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [pos_score_50], Original ATen: [aten._softmax]
        triton_red_fused__softmax_9.run(buf209, arg99_1, buf210, buf211, 25088, 196, grid=grid(25088), stream=stream0)
        buf212 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_32, sub_26, patch_score_33, mul_51, sigmoid_33, pos_score_50, mul_52, attn_54], Original ATen: [aten.sigmoid, aten.rsub, aten._softmax, aten.mul, aten.add]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_10.run(arg100_1, buf203, buf204, buf205, buf209, arg99_1, buf210, buf211, buf212, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg100_1
        del arg99_1
        buf215 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [sum_17, attn_55], Original ATen: [aten.sum, aten.div]
        triton_red_fused_div_sum_11.run(buf212, buf215, 25088, 196, grid=grid(25088), stream=stream0)
        buf214 = reinterpret_tensor(buf202, (1568, 768), (768, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [linear_107], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (1568, 768), (768, 1), 0), reinterpret_tensor(arg101_1, (768, 768), (1, 768), 0), out=buf214)
        del arg101_1
        buf216 = reinterpret_tensor(buf199, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [matmul_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf214, buf216, 1204224, grid=grid(1204224), stream=stream0)
        buf217 = reinterpret_tensor(buf214, (128, 196, 48), (9408, 48, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [matmul_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf215, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf216, (128, 196, 48), (9408, 48, 1), 0), out=buf217)
        buf218 = reinterpret_tensor(buf216, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [x_230], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf217, buf218, 1204224, grid=grid(1204224), stream=stream0)
        buf219 = reinterpret_tensor(buf217, (1568, 768), (768, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (1568, 768), (768, 1), 0), reinterpret_tensor(arg102_1, (768, 768), (1, 768), 0), out=buf219)
        del arg102_1
        buf220 = reinterpret_tensor(buf219, (8, 196, 768), (150528, 768, 1), 0); del buf219  # reuse
        buf224 = reinterpret_tensor(buf218, (8, 196, 768), (150528, 768, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [x_228, x_233, x_234], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf220, buf188, buf195, arg94_1, arg103_1, arg104_1, arg105_1, buf224, 1568, 768, grid=grid(1568), stream=stream0)
        del arg103_1
        del arg104_1
        del arg105_1
        del arg94_1
        buf225 = reinterpret_tensor(buf194, (1568, 3072), (3072, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf224, (1568, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 3072), (1, 768), 0), out=buf225)
        del arg106_1
        buf226 = reinterpret_tensor(buf225, (8, 196, 3072), (602112, 3072, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf226, arg107_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg107_1
        buf227 = reinterpret_tensor(buf224, (1568, 768), (768, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg108_1, (3072, 768), (1, 3072), 0), out=buf227)
        del arg108_1
        buf231 = reinterpret_tensor(buf195, (8, 196, 768), (150528, 768, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_240, x_241], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf220, buf227, arg109_1, arg110_1, arg111_1, buf231, 1568, 768, grid=grid(1568), stream=stream0)
        del arg110_1
        del arg111_1
        buf232 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [linear_111], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (1568, 768), (768, 1), 0), reinterpret_tensor(arg112_1, (768, 1536), (1, 768), 0), out=buf232)
        del arg112_1
        buf233 = reinterpret_tensor(buf188, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [matmul_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf232, buf233, 1204224, grid=grid(1204224), stream=stream0)
        buf234 = reinterpret_tensor(buf201, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [matmul_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf232, buf234, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf235 = reinterpret_tensor(buf209, (128, 196, 196), (38416, 196, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [matmul_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf233, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf234, (128, 48, 196), (9408, 196, 1), 0), out=buf235)
        buf236 = reinterpret_tensor(buf211, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf211  # reuse
        buf237 = reinterpret_tensor(buf210, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [patch_score_35], Original ATen: [aten._softmax]
        triton_per_fused__softmax_5.run(buf235, buf236, buf237, 25088, 196, grid=grid(25088), stream=stream0)
        buf238 = empty_strided_cuda((1, 196, 196, 3), (115248, 588, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rel_indices_7, setitem_21, setitem_22, setitem_23, to_7], Original ATen: [aten.zeros, aten.copy, aten._to_copy]
        triton_poi_fused__to_copy_copy_zeros_6.run(buf238, 115248, grid=grid(115248), stream=stream0)
        buf239 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [linear_112], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf238, buf239, 921984, grid=grid(921984), stream=stream0)
        buf240 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [linear_112], Original ATen: [aten.mm]
        triton_poi_fused_mm_8.run(buf239, buf240, 921984, grid=grid(921984), stream=stream0)
        buf241 = reinterpret_tensor(buf203, (307328, 16), (16, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [linear_112], Original ATen: [aten.mm]
        extern_kernels.mm(buf240, reinterpret_tensor(arg113_1, (3, 16), (1, 3), 0), out=buf241)
        del arg113_1
        buf242 = reinterpret_tensor(buf205, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf205  # reuse
        buf243 = reinterpret_tensor(buf204, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [pos_score_53], Original ATen: [aten._softmax]
        triton_red_fused__softmax_9.run(buf241, arg114_1, buf242, buf243, 25088, 196, grid=grid(25088), stream=stream0)
        buf244 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_34, sub_27, patch_score_35, mul_54, sigmoid_35, pos_score_53, mul_55, attn_57], Original ATen: [aten.sigmoid, aten.rsub, aten._softmax, aten.mul, aten.add]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_10.run(arg115_1, buf235, buf236, buf237, buf241, arg114_1, buf242, buf243, buf244, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg114_1
        del arg115_1
        buf247 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [sum_18, attn_58], Original ATen: [aten.sum, aten.div]
        triton_red_fused_div_sum_11.run(buf244, buf247, 25088, 196, grid=grid(25088), stream=stream0)
        buf246 = reinterpret_tensor(buf234, (1568, 768), (768, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [linear_113], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (1568, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 768), (1, 768), 0), out=buf246)
        del arg116_1
        buf248 = reinterpret_tensor(buf231, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [matmul_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf246, buf248, 1204224, grid=grid(1204224), stream=stream0)
        buf249 = reinterpret_tensor(buf246, (128, 196, 48), (9408, 48, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [matmul_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf247, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf248, (128, 196, 48), (9408, 48, 1), 0), out=buf249)
        buf250 = reinterpret_tensor(buf248, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [x_242], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf249, buf250, 1204224, grid=grid(1204224), stream=stream0)
        buf251 = reinterpret_tensor(buf249, (1568, 768), (768, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (1568, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 768), (1, 768), 0), out=buf251)
        del arg117_1
        buf252 = reinterpret_tensor(buf251, (8, 196, 768), (150528, 768, 1), 0); del buf251  # reuse
        buf256 = reinterpret_tensor(buf250, (8, 196, 768), (150528, 768, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [x_240, x_245, x_246], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf252, buf220, buf227, arg109_1, arg118_1, arg119_1, arg120_1, buf256, 1568, 768, grid=grid(1568), stream=stream0)
        del arg109_1
        del arg118_1
        del arg119_1
        del arg120_1
        buf257 = reinterpret_tensor(buf226, (1568, 3072), (3072, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (1568, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 3072), (1, 768), 0), out=buf257)
        del arg121_1
        buf258 = reinterpret_tensor(buf257, (8, 196, 3072), (602112, 3072, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [x_248], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf258, arg122_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg122_1
        buf259 = reinterpret_tensor(buf256, (1568, 768), (768, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg123_1, (3072, 768), (1, 3072), 0), out=buf259)
        del arg123_1
        buf263 = reinterpret_tensor(buf227, (8, 196, 768), (150528, 768, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [x_252, x_253], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf252, buf259, arg124_1, arg125_1, arg126_1, buf263, 1568, 768, grid=grid(1568), stream=stream0)
        del arg125_1
        del arg126_1
        buf264 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [linear_117], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (1568, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 1536), (1, 768), 0), out=buf264)
        del arg127_1
        buf265 = reinterpret_tensor(buf220, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [matmul_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf264, buf265, 1204224, grid=grid(1204224), stream=stream0)
        buf266 = reinterpret_tensor(buf233, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [matmul_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf264, buf266, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf267 = reinterpret_tensor(buf241, (128, 196, 196), (38416, 196, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [matmul_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf266, (128, 48, 196), (9408, 196, 1), 0), out=buf267)
        buf268 = reinterpret_tensor(buf243, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf243  # reuse
        buf269 = reinterpret_tensor(buf242, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [patch_score_37], Original ATen: [aten._softmax]
        triton_per_fused__softmax_5.run(buf267, buf268, buf269, 25088, 196, grid=grid(25088), stream=stream0)
        buf270 = empty_strided_cuda((1, 196, 196, 3), (115248, 588, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rel_indices_8, setitem_24, setitem_25, setitem_26, to_8], Original ATen: [aten.zeros, aten.copy, aten._to_copy]
        triton_poi_fused__to_copy_copy_zeros_6.run(buf270, 115248, grid=grid(115248), stream=stream0)
        buf271 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [linear_118], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf270, buf271, 921984, grid=grid(921984), stream=stream0)
        buf272 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [linear_118], Original ATen: [aten.mm]
        triton_poi_fused_mm_8.run(buf271, buf272, 921984, grid=grid(921984), stream=stream0)
        buf273 = reinterpret_tensor(buf235, (307328, 16), (16, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [linear_118], Original ATen: [aten.mm]
        extern_kernels.mm(buf272, reinterpret_tensor(arg128_1, (3, 16), (1, 3), 0), out=buf273)
        del arg128_1
        buf274 = reinterpret_tensor(buf237, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf237  # reuse
        buf275 = reinterpret_tensor(buf236, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [pos_score_56], Original ATen: [aten._softmax]
        triton_red_fused__softmax_9.run(buf273, arg129_1, buf274, buf275, 25088, 196, grid=grid(25088), stream=stream0)
        buf276 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_36, sub_28, patch_score_37, mul_57, sigmoid_37, pos_score_56, mul_58, attn_60], Original ATen: [aten.sigmoid, aten.rsub, aten._softmax, aten.mul, aten.add]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_10.run(arg130_1, buf267, buf268, buf269, buf273, arg129_1, buf274, buf275, buf276, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg129_1
        del arg130_1
        buf279 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [sum_19, attn_61], Original ATen: [aten.sum, aten.div]
        triton_red_fused_div_sum_11.run(buf276, buf279, 25088, 196, grid=grid(25088), stream=stream0)
        buf278 = reinterpret_tensor(buf266, (1568, 768), (768, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [linear_119], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (1568, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 768), (1, 768), 0), out=buf278)
        del arg131_1
        buf280 = reinterpret_tensor(buf263, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [matmul_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf278, buf280, 1204224, grid=grid(1204224), stream=stream0)
        buf281 = reinterpret_tensor(buf278, (128, 196, 48), (9408, 48, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [matmul_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf279, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf280, (128, 196, 48), (9408, 48, 1), 0), out=buf281)
        buf282 = reinterpret_tensor(buf280, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [x_254], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf281, buf282, 1204224, grid=grid(1204224), stream=stream0)
        buf283 = reinterpret_tensor(buf281, (1568, 768), (768, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (1568, 768), (768, 1), 0), reinterpret_tensor(arg132_1, (768, 768), (1, 768), 0), out=buf283)
        del arg132_1
        buf284 = reinterpret_tensor(buf283, (8, 196, 768), (150528, 768, 1), 0); del buf283  # reuse
        buf288 = reinterpret_tensor(buf282, (8, 196, 768), (150528, 768, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [x_252, x_257, x_258], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf284, buf252, buf259, arg124_1, arg133_1, arg134_1, arg135_1, buf288, 1568, 768, grid=grid(1568), stream=stream0)
        del arg124_1
        del arg133_1
        del arg134_1
        del arg135_1
        buf289 = reinterpret_tensor(buf258, (1568, 3072), (3072, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf288, (1568, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 3072), (1, 768), 0), out=buf289)
        del arg136_1
        buf290 = reinterpret_tensor(buf289, (8, 196, 3072), (602112, 3072, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [x_260], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf290, arg137_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg137_1
        buf291 = reinterpret_tensor(buf288, (1568, 768), (768, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf290, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg138_1, (3072, 768), (1, 3072), 0), out=buf291)
        del arg138_1
        buf295 = reinterpret_tensor(buf259, (8, 196, 768), (150528, 768, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [x_264, x_265], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_16.run(buf284, buf291, arg139_1, arg140_1, arg141_1, buf295, 1568, 768, grid=grid(1568), stream=stream0)
        del arg140_1
        del arg141_1
        buf296 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [linear_123], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (1568, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 1536), (1, 768), 0), out=buf296)
        del arg142_1
        buf297 = reinterpret_tensor(buf252, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [matmul_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf296, buf297, 1204224, grid=grid(1204224), stream=stream0)
        buf298 = reinterpret_tensor(buf265, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [matmul_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf296, buf298, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del buf296
        buf299 = reinterpret_tensor(buf273, (128, 196, 196), (38416, 196, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [matmul_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf297, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf298, (128, 48, 196), (9408, 196, 1), 0), out=buf299)
        del buf297
        buf300 = reinterpret_tensor(buf275, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf275  # reuse
        buf301 = reinterpret_tensor(buf274, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [patch_score_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_5.run(buf299, buf300, buf301, 25088, 196, grid=grid(25088), stream=stream0)
        buf302 = empty_strided_cuda((1, 196, 196, 3), (115248, 588, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rel_indices_9, setitem_27, setitem_28, setitem_29, to_9], Original ATen: [aten.zeros, aten.copy, aten._to_copy]
        triton_poi_fused__to_copy_copy_zeros_6.run(buf302, 115248, grid=grid(115248), stream=stream0)
        buf303 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf302, buf303, 921984, grid=grid(921984), stream=stream0)
        buf304 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.mm]
        triton_poi_fused_mm_8.run(buf303, buf304, 921984, grid=grid(921984), stream=stream0)
        del buf303
        buf305 = reinterpret_tensor(buf267, (307328, 16), (16, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.mm]
        extern_kernels.mm(buf304, reinterpret_tensor(arg143_1, (3, 16), (1, 3), 0), out=buf305)
        del arg143_1
        del buf304
        buf306 = reinterpret_tensor(buf269, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf269  # reuse
        buf307 = reinterpret_tensor(buf268, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [pos_score_59], Original ATen: [aten._softmax]
        triton_red_fused__softmax_9.run(buf305, arg144_1, buf306, buf307, 25088, 196, grid=grid(25088), stream=stream0)
        buf308 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_38, sub_29, patch_score_39, mul_60, sigmoid_39, pos_score_59, mul_61, attn_63], Original ATen: [aten.sigmoid, aten.rsub, aten._softmax, aten.mul, aten.add]
        triton_poi_fused__softmax_add_mul_rsub_sigmoid_10.run(arg145_1, buf299, buf300, buf301, buf305, arg144_1, buf306, buf307, buf308, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del arg144_1
        del arg145_1
        del buf299
        del buf300
        del buf301
        del buf305
        del buf306
        del buf307
        buf311 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [sum_20, attn_64], Original ATen: [aten.sum, aten.div]
        triton_red_fused_div_sum_11.run(buf308, buf311, 25088, 196, grid=grid(25088), stream=stream0)
        del buf308
        buf310 = reinterpret_tensor(buf298, (1568, 768), (768, 1), 0); del buf298  # reuse
        # Topologically Sorted Source Nodes: [linear_125], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (1568, 768), (768, 1), 0), reinterpret_tensor(arg146_1, (768, 768), (1, 768), 0), out=buf310)
        del arg146_1
        buf312 = reinterpret_tensor(buf295, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [matmul_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf310, buf312, 1204224, grid=grid(1204224), stream=stream0)
        buf313 = reinterpret_tensor(buf310, (128, 196, 48), (9408, 48, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [matmul_43], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf311, (128, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf312, (128, 196, 48), (9408, 48, 1), 0), out=buf313)
        del buf311
        buf314 = reinterpret_tensor(buf312, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [x_266], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf313, buf314, 1204224, grid=grid(1204224), stream=stream0)
        buf315 = reinterpret_tensor(buf313, (1568, 768), (768, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf314, (1568, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 768), (1, 768), 0), out=buf315)
        del arg147_1
        buf316 = reinterpret_tensor(buf315, (8, 196, 768), (150528, 768, 1), 0); del buf315  # reuse
        buf320 = reinterpret_tensor(buf314, (8, 196, 768), (150528, 768, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [x_264, x_269, x_270], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf316, buf284, buf291, arg139_1, arg148_1, arg149_1, arg150_1, buf320, 1568, 768, grid=grid(1568), stream=stream0)
        del arg139_1
        del arg148_1
        del arg149_1
        del arg150_1
        del buf284
        del buf291
        buf321 = reinterpret_tensor(buf290, (1568, 3072), (3072, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf320, (1568, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 3072), (1, 768), 0), out=buf321)
        del arg151_1
        buf322 = reinterpret_tensor(buf321, (8, 196, 3072), (602112, 3072, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [x_272], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf322, arg152_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg152_1
        buf323 = reinterpret_tensor(buf320, (1568, 768), (768, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf322, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg153_1, (3072, 768), (1, 3072), 0), out=buf323)
        del arg153_1
        del buf322
        buf328 = empty_strided_cuda((8, 197, 768), (151296, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_277, x_278], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_18.run(arg4_1, buf316, buf323, arg154_1, arg155_1, arg156_1, buf328, 1576, 768, grid=grid(1576), stream=stream0)
        del arg155_1
        del arg156_1
        buf329 = empty_strided_cuda((1576, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_129], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf328, (1576, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 2304), (1, 768), 0), out=buf329)
        del arg157_1
        buf330 = empty_strided_cuda((8, 16, 197, 48), (151552, 9472, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf329, buf330, 1210368, grid=grid(1210368), stream=stream0)
        buf331 = empty_strided_cuda((8, 16, 48, 197), (151552, 9472, 197, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf329, buf331, 6144, 197, grid=grid(6144, 197), stream=stream0)
        buf332 = empty_strided_cuda((128, 197, 197), (38809, 197, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf330, (128, 197, 48), (9472, 48, 1), 0), reinterpret_tensor(buf331, (128, 48, 197), (9472, 197, 1), 0), out=buf332)
        buf335 = empty_strided_cuda((8, 16, 197, 197), (621056, 38816, 197, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_67], Original ATen: [aten._softmax]
        triton_red_fused__softmax_21.run(buf332, buf335, 25216, 197, grid=grid(25216), stream=stream0)
        buf336 = reinterpret_tensor(buf331, (8, 16, 197, 48), (151552, 9472, 48, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [matmul_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf329, buf336, 1210368, grid=grid(1210368), stream=stream0)
        buf337 = reinterpret_tensor(buf328, (128, 197, 48), (9456, 48, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [matmul_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (128, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf336, (128, 197, 48), (9472, 48, 1), 0), out=buf337)
        buf338 = empty_strided_cuda((8, 197, 16, 48), (151296, 768, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_279], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf337, buf338, 1210368, grid=grid(1210368), stream=stream0)
        buf339 = reinterpret_tensor(buf337, (1576, 768), (768, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf338, (1576, 768), (768, 1), 0), reinterpret_tensor(arg158_1, (768, 768), (1, 768), 0), out=buf339)
        del arg158_1
        buf340 = reinterpret_tensor(buf339, (8, 197, 768), (151296, 768, 1), 0); del buf339  # reuse
        buf344 = reinterpret_tensor(buf338, (8, 197, 768), (151296, 768, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [x_277, x_282, x_283], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_24.run(buf340, arg4_1, buf316, buf323, arg154_1, arg159_1, arg160_1, arg161_1, buf344, 1576, 768, grid=grid(1576), stream=stream0)
        del arg154_1
        del arg159_1
        del arg160_1
        del arg161_1
        del arg4_1
        del buf316
        del buf323
        buf345 = empty_strided_cuda((1576, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf344, (1576, 768), (768, 1), 0), reinterpret_tensor(arg162_1, (768, 3072), (1, 768), 0), out=buf345)
        del arg162_1
        buf346 = reinterpret_tensor(buf345, (8, 197, 3072), (605184, 3072, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [x_285], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf346, arg163_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg163_1
        buf347 = reinterpret_tensor(buf344, (1576, 768), (768, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg164_1, (3072, 768), (1, 3072), 0), out=buf347)
        del arg164_1
        buf351 = empty_strided_cuda((8, 197, 768), (151296, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_289, x_290], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf340, buf347, arg165_1, arg166_1, arg167_1, buf351, 1576, 768, grid=grid(1576), stream=stream0)
        del arg166_1
        del arg167_1
        buf352 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [linear_133], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf351, (1576, 768), (768, 1), 0), reinterpret_tensor(arg168_1, (768, 2304), (1, 768), 0), out=buf352)
        del arg168_1
        buf353 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [matmul_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf352, buf353, 1210368, grid=grid(1210368), stream=stream0)
        buf354 = reinterpret_tensor(buf330, (8, 16, 48, 197), (151552, 9472, 197, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [matmul_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf352, buf354, 6144, 197, grid=grid(6144, 197), stream=stream0)
        buf355 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [matmul_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf353, (128, 197, 48), (9472, 48, 1), 0), reinterpret_tensor(buf354, (128, 48, 197), (9472, 197, 1), 0), out=buf355)
        del buf353
        buf358 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [attn_70], Original ATen: [aten._softmax]
        triton_red_fused__softmax_21.run(buf355, buf358, 25216, 197, grid=grid(25216), stream=stream0)
        del buf355
        buf359 = reinterpret_tensor(buf354, (8, 16, 197, 48), (151552, 9472, 48, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [matmul_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf352, buf359, 1210368, grid=grid(1210368), stream=stream0)
        del buf352
        buf360 = reinterpret_tensor(buf351, (128, 197, 48), (9456, 48, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [matmul_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf358, (128, 197, 197), (38816, 197, 1), 0), reinterpret_tensor(buf359, (128, 197, 48), (9472, 48, 1), 0), out=buf360)
        del buf358
        del buf359
        buf361 = empty_strided_cuda((8, 197, 16, 48), (151296, 768, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf360, buf361, 1210368, grid=grid(1210368), stream=stream0)
        buf362 = reinterpret_tensor(buf360, (1576, 768), (768, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf361, (1576, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 768), (1, 768), 0), out=buf362)
        del arg169_1
        buf363 = reinterpret_tensor(buf362, (8, 197, 768), (151296, 768, 1), 0); del buf362  # reuse
        buf367 = reinterpret_tensor(buf361, (8, 197, 768), (151296, 768, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [x_289, x_294, x_295], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf363, buf340, buf347, arg165_1, arg170_1, arg171_1, arg172_1, buf367, 1576, 768, grid=grid(1576), stream=stream0)
        del arg165_1
        del arg170_1
        del arg171_1
        del arg172_1
        del buf340
        del buf347
        buf368 = reinterpret_tensor(buf346, (1576, 3072), (3072, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf367, (1576, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 3072), (1, 768), 0), out=buf368)
        del arg173_1
        buf369 = reinterpret_tensor(buf368, (8, 197, 3072), (605184, 3072, 1), 0); del buf368  # reuse
        # Topologically Sorted Source Nodes: [x_297], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf369, arg174_1, 4841472, grid=grid(4841472), stream=stream0)
        del arg174_1
        buf370 = reinterpret_tensor(buf367, (1576, 768), (768, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf369, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg175_1, (3072, 768), (1, 3072), 0), out=buf370)
        del arg175_1
        del buf369
        buf371 = empty_strided_cuda((8, 197, 1), (197, 1, 1600), torch.float32)
        buf372 = empty_strided_cuda((8, 197, 1), (197, 1, 1600), torch.float32)
        # Topologically Sorted Source Nodes: [x_301, x_302], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_28.run(buf363, buf370, arg176_1, buf371, buf372, 1576, 768, grid=grid(1576), stream=stream0)
        buf374 = empty_strided_cuda((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_304], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf363, buf370, arg176_1, buf371, buf372, arg177_1, arg178_1, buf374, 6144, grid=grid(6144), stream=stream0)
        del arg176_1
        del arg177_1
        del arg178_1
        del buf363
        del buf370
        del buf371
        del buf372
        buf375 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_304, x_305], Original ATen: [aten.clone, aten.addmm]
        extern_kernels.addmm(arg180_1, buf374, reinterpret_tensor(arg179_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf375)
        del arg179_1
        del arg180_1
        del buf374
    return (buf375, buf14, buf46, buf78, buf110, buf142, buf174, buf206, buf238, buf270, buf302, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
    arg67_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
    arg127_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((16, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convit_base', benchmark_compiled_module)
