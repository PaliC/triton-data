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


# kernel path: /tmp/torchinductor_sahanp/3r/c3rfzkgn2zcpqymubtgpsfwsx3as7ylgbcwloydbwgmlnlfjin5h.py
# Topologically Sorted Source Nodes: [layer_norm_25], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_25 => clone_86, var_mean_25
# Graph fragment:
#   %clone_86 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_74,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_86, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_red_fused_native_layer_norm_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 6
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


# kernel path: /tmp/torchinductor_sahanp/y7/cy7oiuysjut4sphvdg4zyz43oruo6fg7kpos54wvgdrykrckci4p.py
# Topologically Sorted Source Nodes: [layer_norm_25], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_25 => clone_86, var_mean_25
# Graph fragment:
#   %clone_86 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_74,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_86, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_1 = async_compile.triton('triton_per_fused_native_layer_norm_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (1176*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (1176*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (196*r2) + (1176*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/4x/c4xdfacsnnzo2lfyn3bpfxf3zknrxgz7a3jnw73rwnuefhg55264.py
# Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_152 => clone_87
# Graph fragment:
#   %clone_87 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_75,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 196) % 768
    x0 = xindex % 196
    x2 = (xindex // 150528)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 768.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/l6/cl6jrr5i6eq74zcucq2ew4de6tzw25jpbzdkdzs5rxxhw5p6lk5m.py
# Topologically Sorted Source Nodes: [x_152, x_153], Original ATen: [aten.add, aten.gelu]
# Source node to ATen node mapping:
#   x_152 => add_112
#   x_153 => add_113, erf_24, mul_124, mul_125, mul_126
# Graph fragment:
#   %add_112 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_99, %arg6_1), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_112, 0.5), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_112, 0.7071067811865476), kwargs = {})
#   %erf_24 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_125,), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_24, 1), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %add_113), kwargs = {})
triton_poi_fused_add_gelu_3 = async_compile.triton('triton_poi_fused_add_gelu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_gelu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
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


# kernel path: /tmp/torchinductor_sahanp/kv/ckv7fbsycsz3iis4oyfdcagdtywmw2rspvryinljfw56cjk6fkzs.py
# Topologically Sorted Source Nodes: [x_157, layer_norm_26], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_26 => clone_90, var_mean_26
#   x_157 => add_114
# Graph fragment:
#   %add_114 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_74, %permute_78), kwargs = {})
#   %clone_90 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_114,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_90, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_4 = async_compile.triton('triton_red_fused_add_native_layer_norm_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 6
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r3) + (25088*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (196*r3) + (25088*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/y4/cy44i2e7sfj3dr362ot5dwnkdtdy6nfdgnodu4p7zt273nf6fbu7.py
# Topologically Sorted Source Nodes: [x_157, layer_norm_26], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_26 => add_115, add_116, clone_90, mul_127, mul_128, rsqrt_26, sub_26, var_mean_26
#   x_157 => add_114
# Graph fragment:
#   %add_114 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_74, %permute_78), kwargs = {})
#   %clone_90 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_114,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_90, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_90, %getitem_53), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_52, 1e-06), kwargs = {})
#   %rsqrt_26 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_115,), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %rsqrt_26), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, %arg9_1), kwargs = {})
#   %add_116 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %arg10_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_5 = async_compile.triton('triton_poi_fused_add_native_layer_norm_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 768.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (y0 + (768*x2) + (150528*y1)), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2x/c2xwst7oeeuizeyujf5vhp46eeosfibfsuha5yrv4o37aobllqva.py
# Topologically Sorted Source Nodes: [x_159], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_159 => add_117, erf_25, mul_129, mul_130, mul_131
# Graph fragment:
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_103, 0.5), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_103, 0.7071067811865476), kwargs = {})
#   %erf_25 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_130,), kwargs = {})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_25, 1), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_129, %add_117), kwargs = {})
triton_poi_fused_gelu_6 = async_compile.triton('triton_poi_fused_gelu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/eu/ceuqhwxh7no6pvb525wu6zx7zhomi3u5rw77yf7cqogcv7cxtcnh.py
# Topologically Sorted Source Nodes: [x_157, x_163], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_157 => add_114
#   x_163 => add_118
# Graph fragment:
#   %add_114 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_74, %permute_78), kwargs = {})
#   %add_118 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_114, %view_105), kwargs = {})
triton_poi_fused_add_7 = async_compile.triton('triton_poi_fused_add_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (768*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/26/c26llusxsdh3m5ookj3vcmsgrg7o6stsij6gl7rmbgj4jc2sbii5.py
# Topologically Sorted Source Nodes: [layer_norm_27], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_27 => clone_93, var_mean_27
# Graph fragment:
#   %clone_93 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_118,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_93, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_layer_norm_8 = async_compile.triton('triton_red_fused_native_layer_norm_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_8(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eq/ceqbccievndvibxhsqn7tx6slctucapywwgzvsbfavniwjri2kwk.py
# Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_164 => clone_94
# Graph fragment:
#   %clone_94 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_81,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_poi_fused_clone_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 196
    x2 = (xindex // 150528)
    x1 = (xindex // 196) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2c/c2cr2nsq7wlc5fbwtw57w7kurt7fljlsonjeguobcmapomjp3yje.py
# Topologically Sorted Source Nodes: [x_169, layer_norm_28], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_28 => clone_97, var_mean_28
#   x_169 => add_123
# Graph fragment:
#   %add_123 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_118, %permute_84), kwargs = {})
#   %clone_97 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_123,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_28 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_97, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_10 = async_compile.triton('triton_red_fused_add_native_layer_norm_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wj/cwj4ro5hl726cn43jboqebja2vyfxabkg3ej4oyjage7nebqxrfr.py
# Topologically Sorted Source Nodes: [x_169, layer_norm_28], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_28 => add_124, add_125, clone_97, mul_137, mul_138, rsqrt_28, sub_28, var_mean_28
#   x_169 => add_123
# Graph fragment:
#   %add_123 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_118, %permute_84), kwargs = {})
#   %clone_97 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_123,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_28 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_97, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_97, %getitem_57), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_56, 1e-06), kwargs = {})
#   %rsqrt_28 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_124,), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_28), kwargs = {})
#   %mul_138 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_137, %arg21_1), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_138, %arg22_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_11 = async_compile.triton('triton_poi_fused_add_native_layer_norm_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 768)
    y0 = yindex % 768
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (768*x2) + (150528*y1)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/e3/ce3o6ar4445cwvda6czhz2piapjpfb6zyai6zkikkg3dmcy46dwj.py
# Topologically Sorted Source Nodes: [x_169, x_175], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_169 => add_123
#   x_175 => add_127
# Graph fragment:
#   %add_123 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_118, %permute_84), kwargs = {})
#   %add_127 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_123, %view_113), kwargs = {})
triton_poi_fused_add_12 = async_compile.triton('triton_poi_fused_add_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (768*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ct/cctljdv6b62yk4qwdivicbzgr3n4a4bbe3sayffn5k2ji5z6qffy.py
# Topologically Sorted Source Nodes: [x_296, x_297], Original ATen: [aten.native_layer_norm, aten.mean]
# Source node to ATen node mapping:
#   x_296 => add_218, add_219, mul_242, mul_243, rsqrt_49, sub_49, var_mean_49
#   x_297 => mean_1
# Graph fragment:
#   %var_mean_49 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_170, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_170, %getitem_99), kwargs = {})
#   %add_218 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_98, 1e-06), kwargs = {})
#   %rsqrt_49 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_218,), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %rsqrt_49), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_242, %arg147_1), kwargs = {})
#   %add_219 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_243, %arg148_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_219, [1]), kwargs = {})
triton_per_fused_mean_native_layer_norm_13 = async_compile.triton('triton_per_fused_mean_native_layer_norm_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_native_layer_norm_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 768)
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (196*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (196*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = 196.0
    tmp19 = tmp17 / tmp18
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (384, 196), (196, 1))
    assert_size_stride(arg6_1, (384, ), (1, ))
    assert_size_stride(arg7_1, (196, 384), (384, 1))
    assert_size_stride(arg8_1, (196, ), (1, ))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (3072, 768), (768, 1))
    assert_size_stride(arg12_1, (3072, ), (1, ))
    assert_size_stride(arg13_1, (768, 3072), (3072, 1))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (384, 196), (196, 1))
    assert_size_stride(arg18_1, (384, ), (1, ))
    assert_size_stride(arg19_1, (196, 384), (384, 1))
    assert_size_stride(arg20_1, (196, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (3072, 768), (768, 1))
    assert_size_stride(arg24_1, (3072, ), (1, ))
    assert_size_stride(arg25_1, (768, 3072), (3072, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (384, 196), (196, 1))
    assert_size_stride(arg30_1, (384, ), (1, ))
    assert_size_stride(arg31_1, (196, 384), (384, 1))
    assert_size_stride(arg32_1, (196, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (3072, 768), (768, 1))
    assert_size_stride(arg36_1, (3072, ), (1, ))
    assert_size_stride(arg37_1, (768, 3072), (3072, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (384, 196), (196, 1))
    assert_size_stride(arg42_1, (384, ), (1, ))
    assert_size_stride(arg43_1, (196, 384), (384, 1))
    assert_size_stride(arg44_1, (196, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (3072, 768), (768, 1))
    assert_size_stride(arg48_1, (3072, ), (1, ))
    assert_size_stride(arg49_1, (768, 3072), (3072, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (384, 196), (196, 1))
    assert_size_stride(arg54_1, (384, ), (1, ))
    assert_size_stride(arg55_1, (196, 384), (384, 1))
    assert_size_stride(arg56_1, (196, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (3072, 768), (768, 1))
    assert_size_stride(arg60_1, (3072, ), (1, ))
    assert_size_stride(arg61_1, (768, 3072), (3072, 1))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (384, 196), (196, 1))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (196, 384), (384, 1))
    assert_size_stride(arg68_1, (196, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (3072, 768), (768, 1))
    assert_size_stride(arg72_1, (3072, ), (1, ))
    assert_size_stride(arg73_1, (768, 3072), (3072, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (384, 196), (196, 1))
    assert_size_stride(arg78_1, (384, ), (1, ))
    assert_size_stride(arg79_1, (196, 384), (384, 1))
    assert_size_stride(arg80_1, (196, ), (1, ))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (3072, 768), (768, 1))
    assert_size_stride(arg84_1, (3072, ), (1, ))
    assert_size_stride(arg85_1, (768, 3072), (3072, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (384, 196), (196, 1))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (196, 384), (384, 1))
    assert_size_stride(arg92_1, (196, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (3072, 768), (768, 1))
    assert_size_stride(arg96_1, (3072, ), (1, ))
    assert_size_stride(arg97_1, (768, 3072), (3072, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (384, 196), (196, 1))
    assert_size_stride(arg102_1, (384, ), (1, ))
    assert_size_stride(arg103_1, (196, 384), (384, 1))
    assert_size_stride(arg104_1, (196, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (3072, 768), (768, 1))
    assert_size_stride(arg108_1, (3072, ), (1, ))
    assert_size_stride(arg109_1, (768, 3072), (3072, 1))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (384, 196), (196, 1))
    assert_size_stride(arg114_1, (384, ), (1, ))
    assert_size_stride(arg115_1, (196, 384), (384, 1))
    assert_size_stride(arg116_1, (196, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (3072, 768), (768, 1))
    assert_size_stride(arg120_1, (3072, ), (1, ))
    assert_size_stride(arg121_1, (768, 3072), (3072, 1))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (384, 196), (196, 1))
    assert_size_stride(arg126_1, (384, ), (1, ))
    assert_size_stride(arg127_1, (196, 384), (384, 1))
    assert_size_stride(arg128_1, (196, ), (1, ))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (3072, 768), (768, 1))
    assert_size_stride(arg132_1, (3072, ), (1, ))
    assert_size_stride(arg133_1, (768, 3072), (3072, 1))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (384, 196), (196, 1))
    assert_size_stride(arg138_1, (384, ), (1, ))
    assert_size_stride(arg139_1, (196, 384), (384, 1))
    assert_size_stride(arg140_1, (196, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (3072, 768), (768, 1))
    assert_size_stride(arg144_1, (3072, ), (1, ))
    assert_size_stride(arg145_1, (768, 3072), (3072, 1))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (1000, 768), (768, 1))
    assert_size_stride(arg150_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((8, 196, 1, 6), (1176, 1, 9408, 196), torch.float32)
        buf2 = empty_strided_cuda((8, 196, 1, 6), (1176, 1, 9408, 196), torch.float32)
        buf3 = empty_strided_cuda((8, 196, 1, 6), (1176, 1, 9408, 196), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_25], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf0, arg2_1, buf1, buf2, buf3, 9408, 128, grid=grid(9408), stream=stream0)
        buf4 = empty_strided_cuda((8, 196, 1), (196, 1, 1568), torch.float32)
        buf5 = empty_strided_cuda((8, 196, 1), (196, 1, 1568), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_25], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 1568, 6, grid=grid(1568), stream=stream0)
        buf7 = empty_strided_cuda((8, 768, 196), (150528, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf0, arg2_1, buf4, buf5, arg3_1, arg4_1, buf7, 1204224, grid=grid(1204224), stream=stream0)
        del arg3_1
        del arg4_1
        buf8 = empty_strided_cuda((6144, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (6144, 196), (196, 1), 0), reinterpret_tensor(arg5_1, (196, 384), (1, 196), 0), out=buf8)
        del arg5_1
        buf9 = reinterpret_tensor(buf8, (8, 768, 384), (294912, 384, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_152, x_153], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf9, arg6_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg6_1
        buf10 = reinterpret_tensor(buf7, (6144, 196), (196, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf9, (6144, 384), (384, 1), 0), reinterpret_tensor(arg7_1, (384, 196), (1, 384), 0), out=buf10)
        del arg7_1
        buf11 = buf3; del buf3  # reuse
        buf12 = buf2; del buf2  # reuse
        buf13 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_157, layer_norm_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf0, arg2_1, buf10, arg8_1, buf11, buf12, buf13, 9408, 128, grid=grid(9408), stream=stream0)
        buf14 = buf5; del buf5  # reuse
        buf15 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_157, layer_norm_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf11, buf12, buf13, buf14, buf15, 1568, 6, grid=grid(1568), stream=stream0)
        buf17 = empty_strided_cuda((8, 196, 768), (150528, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_157, layer_norm_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_5.run(buf0, arg2_1, buf10, arg8_1, buf14, buf15, arg9_1, arg10_1, buf17, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg10_1
        del arg9_1
        buf18 = empty_strided_cuda((1568, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf17, (1568, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 3072), (1, 768), 0), out=buf18)
        del arg11_1
        buf19 = reinterpret_tensor(buf18, (8, 196, 3072), (602112, 3072, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_159], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf19, arg12_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg12_1
        buf20 = reinterpret_tensor(buf17, (1568, 768), (768, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg13_1, (3072, 768), (1, 3072), 0), out=buf20)
        del arg13_1
        buf21 = reinterpret_tensor(buf0, (8, 196, 768), (150528, 1, 196), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_157, x_163], Original ATen: [aten.add]
        triton_poi_fused_add_7.run(buf21, arg2_1, buf10, arg8_1, buf20, arg14_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg14_1
        del arg2_1
        del arg8_1
        buf22 = buf13; del buf13  # reuse
        buf23 = buf12; del buf12  # reuse
        buf24 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_27], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf21, buf22, buf23, buf24, 9408, 128, grid=grid(9408), stream=stream0)
        buf25 = buf15; del buf15  # reuse
        buf26 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_27], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf22, buf23, buf24, buf25, buf26, 1568, 6, grid=grid(1568), stream=stream0)
        buf28 = reinterpret_tensor(buf20, (8, 768, 196), (150528, 196, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf21, buf25, buf26, arg15_1, arg16_1, buf28, 1204224, grid=grid(1204224), stream=stream0)
        del arg15_1
        del arg16_1
        buf29 = reinterpret_tensor(buf9, (6144, 384), (384, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [x_164], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (6144, 196), (196, 1), 0), reinterpret_tensor(arg17_1, (196, 384), (1, 196), 0), out=buf29)
        del arg17_1
        buf30 = reinterpret_tensor(buf29, (8, 768, 384), (294912, 384, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_164, x_165], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf30, arg18_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg18_1
        buf31 = reinterpret_tensor(buf28, (6144, 196), (196, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (6144, 384), (384, 1), 0), reinterpret_tensor(arg19_1, (384, 196), (1, 384), 0), out=buf31)
        del arg19_1
        buf32 = buf24; del buf24  # reuse
        buf33 = buf23; del buf23  # reuse
        buf34 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_169, layer_norm_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf21, buf31, arg20_1, buf32, buf33, buf34, 9408, 128, grid=grid(9408), stream=stream0)
        buf35 = buf26; del buf26  # reuse
        buf36 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_169, layer_norm_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf32, buf33, buf34, buf35, buf36, 1568, 6, grid=grid(1568), stream=stream0)
        buf38 = reinterpret_tensor(buf10, (8, 196, 768), (150528, 768, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_169, layer_norm_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf21, buf31, arg20_1, buf35, buf36, arg21_1, arg22_1, buf38, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg21_1
        del arg22_1
        buf39 = reinterpret_tensor(buf19, (1568, 3072), (3072, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (1568, 768), (768, 1), 0), reinterpret_tensor(arg23_1, (768, 3072), (1, 768), 0), out=buf39)
        del arg23_1
        buf40 = reinterpret_tensor(buf39, (8, 196, 3072), (602112, 3072, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_171], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf40, arg24_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg24_1
        buf41 = reinterpret_tensor(buf38, (1568, 768), (768, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg25_1, (3072, 768), (1, 3072), 0), out=buf41)
        del arg25_1
        buf42 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_169, x_175], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf42, buf31, arg20_1, buf41, arg26_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg20_1
        del arg26_1
        buf43 = buf34; del buf34  # reuse
        buf44 = buf33; del buf33  # reuse
        buf45 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_29], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf42, buf43, buf44, buf45, 9408, 128, grid=grid(9408), stream=stream0)
        buf46 = buf36; del buf36  # reuse
        buf47 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_29], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf43, buf44, buf45, buf46, buf47, 1568, 6, grid=grid(1568), stream=stream0)
        buf49 = reinterpret_tensor(buf41, (8, 768, 196), (150528, 196, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_176], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf42, buf46, buf47, arg27_1, arg28_1, buf49, 1204224, grid=grid(1204224), stream=stream0)
        del arg27_1
        del arg28_1
        buf50 = reinterpret_tensor(buf30, (6144, 384), (384, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_176], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (6144, 196), (196, 1), 0), reinterpret_tensor(arg29_1, (196, 384), (1, 196), 0), out=buf50)
        del arg29_1
        buf51 = reinterpret_tensor(buf50, (8, 768, 384), (294912, 384, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_176, x_177], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf51, arg30_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg30_1
        buf52 = reinterpret_tensor(buf49, (6144, 196), (196, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (6144, 384), (384, 1), 0), reinterpret_tensor(arg31_1, (384, 196), (1, 384), 0), out=buf52)
        del arg31_1
        buf53 = buf45; del buf45  # reuse
        buf54 = buf44; del buf44  # reuse
        buf55 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_181, layer_norm_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf42, buf52, arg32_1, buf53, buf54, buf55, 9408, 128, grid=grid(9408), stream=stream0)
        buf56 = buf47; del buf47  # reuse
        buf57 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_181, layer_norm_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf53, buf54, buf55, buf56, buf57, 1568, 6, grid=grid(1568), stream=stream0)
        buf59 = reinterpret_tensor(buf31, (8, 196, 768), (150528, 768, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_181, layer_norm_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf42, buf52, arg32_1, buf56, buf57, arg33_1, arg34_1, buf59, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg33_1
        del arg34_1
        buf60 = reinterpret_tensor(buf40, (1568, 3072), (3072, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (1568, 768), (768, 1), 0), reinterpret_tensor(arg35_1, (768, 3072), (1, 768), 0), out=buf60)
        del arg35_1
        buf61 = reinterpret_tensor(buf60, (8, 196, 3072), (602112, 3072, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_183], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf61, arg36_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg36_1
        buf62 = reinterpret_tensor(buf59, (1568, 768), (768, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg37_1, (3072, 768), (1, 3072), 0), out=buf62)
        del arg37_1
        buf63 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_181, x_187], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf63, buf52, arg32_1, buf62, arg38_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg32_1
        del arg38_1
        buf64 = buf55; del buf55  # reuse
        buf65 = buf54; del buf54  # reuse
        buf66 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_31], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf63, buf64, buf65, buf66, 9408, 128, grid=grid(9408), stream=stream0)
        buf67 = buf57; del buf57  # reuse
        buf68 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_31], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf64, buf65, buf66, buf67, buf68, 1568, 6, grid=grid(1568), stream=stream0)
        buf70 = reinterpret_tensor(buf62, (8, 768, 196), (150528, 196, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf63, buf67, buf68, arg39_1, arg40_1, buf70, 1204224, grid=grid(1204224), stream=stream0)
        del arg39_1
        del arg40_1
        buf71 = reinterpret_tensor(buf51, (6144, 384), (384, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (6144, 196), (196, 1), 0), reinterpret_tensor(arg41_1, (196, 384), (1, 196), 0), out=buf71)
        del arg41_1
        buf72 = reinterpret_tensor(buf71, (8, 768, 384), (294912, 384, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_188, x_189], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf72, arg42_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg42_1
        buf73 = reinterpret_tensor(buf70, (6144, 196), (196, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (6144, 384), (384, 1), 0), reinterpret_tensor(arg43_1, (384, 196), (1, 384), 0), out=buf73)
        del arg43_1
        buf74 = buf66; del buf66  # reuse
        buf75 = buf65; del buf65  # reuse
        buf76 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_193, layer_norm_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf63, buf73, arg44_1, buf74, buf75, buf76, 9408, 128, grid=grid(9408), stream=stream0)
        buf77 = buf68; del buf68  # reuse
        buf78 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_193, layer_norm_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf74, buf75, buf76, buf77, buf78, 1568, 6, grid=grid(1568), stream=stream0)
        buf80 = reinterpret_tensor(buf52, (8, 196, 768), (150528, 768, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_193, layer_norm_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf63, buf73, arg44_1, buf77, buf78, arg45_1, arg46_1, buf80, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg45_1
        del arg46_1
        buf81 = reinterpret_tensor(buf61, (1568, 3072), (3072, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (1568, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 3072), (1, 768), 0), out=buf81)
        del arg47_1
        buf82 = reinterpret_tensor(buf81, (8, 196, 3072), (602112, 3072, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_195], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf82, arg48_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg48_1
        buf83 = reinterpret_tensor(buf80, (1568, 768), (768, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg49_1, (3072, 768), (1, 3072), 0), out=buf83)
        del arg49_1
        buf84 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [x_193, x_199], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf84, buf73, arg44_1, buf83, arg50_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg44_1
        del arg50_1
        buf85 = buf76; del buf76  # reuse
        buf86 = buf75; del buf75  # reuse
        buf87 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_33], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf84, buf85, buf86, buf87, 9408, 128, grid=grid(9408), stream=stream0)
        buf88 = buf78; del buf78  # reuse
        buf89 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_33], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf85, buf86, buf87, buf88, buf89, 1568, 6, grid=grid(1568), stream=stream0)
        buf91 = reinterpret_tensor(buf83, (8, 768, 196), (150528, 196, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_200], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf84, buf88, buf89, arg51_1, arg52_1, buf91, 1204224, grid=grid(1204224), stream=stream0)
        del arg51_1
        del arg52_1
        buf92 = reinterpret_tensor(buf72, (6144, 384), (384, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_200], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (6144, 196), (196, 1), 0), reinterpret_tensor(arg53_1, (196, 384), (1, 196), 0), out=buf92)
        del arg53_1
        buf93 = reinterpret_tensor(buf92, (8, 768, 384), (294912, 384, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_200, x_201], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf93, arg54_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg54_1
        buf94 = reinterpret_tensor(buf91, (6144, 196), (196, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (6144, 384), (384, 1), 0), reinterpret_tensor(arg55_1, (384, 196), (1, 384), 0), out=buf94)
        del arg55_1
        buf95 = buf87; del buf87  # reuse
        buf96 = buf86; del buf86  # reuse
        buf97 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_205, layer_norm_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf84, buf94, arg56_1, buf95, buf96, buf97, 9408, 128, grid=grid(9408), stream=stream0)
        buf98 = buf89; del buf89  # reuse
        buf99 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_205, layer_norm_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf95, buf96, buf97, buf98, buf99, 1568, 6, grid=grid(1568), stream=stream0)
        buf101 = reinterpret_tensor(buf73, (8, 196, 768), (150528, 768, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_205, layer_norm_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf84, buf94, arg56_1, buf98, buf99, arg57_1, arg58_1, buf101, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg57_1
        del arg58_1
        buf102 = reinterpret_tensor(buf82, (1568, 3072), (3072, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (1568, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 3072), (1, 768), 0), out=buf102)
        del arg59_1
        buf103 = reinterpret_tensor(buf102, (8, 196, 3072), (602112, 3072, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_207], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf103, arg60_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg60_1
        buf104 = reinterpret_tensor(buf101, (1568, 768), (768, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg61_1, (3072, 768), (1, 3072), 0), out=buf104)
        del arg61_1
        buf105 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_205, x_211], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf105, buf94, arg56_1, buf104, arg62_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg56_1
        del arg62_1
        buf106 = buf97; del buf97  # reuse
        buf107 = buf96; del buf96  # reuse
        buf108 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_35], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf105, buf106, buf107, buf108, 9408, 128, grid=grid(9408), stream=stream0)
        buf109 = buf99; del buf99  # reuse
        buf110 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_35], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf106, buf107, buf108, buf109, buf110, 1568, 6, grid=grid(1568), stream=stream0)
        buf112 = reinterpret_tensor(buf94, (8, 768, 196), (150528, 196, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_212], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf105, buf109, buf110, arg63_1, arg64_1, buf112, 1204224, grid=grid(1204224), stream=stream0)
        del arg63_1
        del arg64_1
        buf113 = reinterpret_tensor(buf93, (6144, 384), (384, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_212], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (6144, 196), (196, 1), 0), reinterpret_tensor(arg65_1, (196, 384), (1, 196), 0), out=buf113)
        del arg65_1
        buf114 = reinterpret_tensor(buf113, (8, 768, 384), (294912, 384, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [x_212, x_213], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf114, arg66_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg66_1
        buf115 = reinterpret_tensor(buf112, (6144, 196), (196, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (6144, 384), (384, 1), 0), reinterpret_tensor(arg67_1, (384, 196), (1, 384), 0), out=buf115)
        del arg67_1
        buf116 = buf108; del buf108  # reuse
        buf117 = buf107; del buf107  # reuse
        buf118 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_217, layer_norm_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf105, buf115, arg68_1, buf116, buf117, buf118, 9408, 128, grid=grid(9408), stream=stream0)
        buf119 = buf110; del buf110  # reuse
        buf120 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_217, layer_norm_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf116, buf117, buf118, buf119, buf120, 1568, 6, grid=grid(1568), stream=stream0)
        buf122 = reinterpret_tensor(buf104, (8, 196, 768), (150528, 768, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_217, layer_norm_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf105, buf115, arg68_1, buf119, buf120, arg69_1, arg70_1, buf122, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg69_1
        del arg70_1
        buf123 = reinterpret_tensor(buf103, (1568, 3072), (3072, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (1568, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 3072), (1, 768), 0), out=buf123)
        del arg71_1
        buf124 = reinterpret_tensor(buf123, (8, 196, 3072), (602112, 3072, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_219], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf124, arg72_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg72_1
        buf125 = reinterpret_tensor(buf122, (1568, 768), (768, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg73_1, (3072, 768), (1, 3072), 0), out=buf125)
        del arg73_1
        buf126 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_217, x_223], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf126, buf115, arg68_1, buf125, arg74_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg68_1
        del arg74_1
        buf127 = buf118; del buf118  # reuse
        buf128 = buf117; del buf117  # reuse
        buf129 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_37], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf126, buf127, buf128, buf129, 9408, 128, grid=grid(9408), stream=stream0)
        buf130 = buf120; del buf120  # reuse
        buf131 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_37], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf127, buf128, buf129, buf130, buf131, 1568, 6, grid=grid(1568), stream=stream0)
        buf133 = reinterpret_tensor(buf125, (8, 768, 196), (150528, 196, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [x_224], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf126, buf130, buf131, arg75_1, arg76_1, buf133, 1204224, grid=grid(1204224), stream=stream0)
        del arg75_1
        del arg76_1
        buf134 = reinterpret_tensor(buf114, (6144, 384), (384, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [x_224], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (6144, 196), (196, 1), 0), reinterpret_tensor(arg77_1, (196, 384), (1, 196), 0), out=buf134)
        del arg77_1
        buf135 = reinterpret_tensor(buf134, (8, 768, 384), (294912, 384, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [x_224, x_225], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf135, arg78_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg78_1
        buf136 = reinterpret_tensor(buf133, (6144, 196), (196, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (6144, 384), (384, 1), 0), reinterpret_tensor(arg79_1, (384, 196), (1, 384), 0), out=buf136)
        del arg79_1
        buf137 = buf129; del buf129  # reuse
        buf138 = buf128; del buf128  # reuse
        buf139 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_229, layer_norm_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf126, buf136, arg80_1, buf137, buf138, buf139, 9408, 128, grid=grid(9408), stream=stream0)
        buf140 = buf131; del buf131  # reuse
        buf141 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_229, layer_norm_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf137, buf138, buf139, buf140, buf141, 1568, 6, grid=grid(1568), stream=stream0)
        buf143 = reinterpret_tensor(buf115, (8, 196, 768), (150528, 768, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_229, layer_norm_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf126, buf136, arg80_1, buf140, buf141, arg81_1, arg82_1, buf143, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg81_1
        del arg82_1
        buf144 = reinterpret_tensor(buf124, (1568, 3072), (3072, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (1568, 768), (768, 1), 0), reinterpret_tensor(arg83_1, (768, 3072), (1, 768), 0), out=buf144)
        del arg83_1
        buf145 = reinterpret_tensor(buf144, (8, 196, 3072), (602112, 3072, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_231], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf145, arg84_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg84_1
        buf146 = reinterpret_tensor(buf143, (1568, 768), (768, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg85_1, (3072, 768), (1, 3072), 0), out=buf146)
        del arg85_1
        buf147 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_229, x_235], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf147, buf136, arg80_1, buf146, arg86_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg80_1
        del arg86_1
        buf148 = buf139; del buf139  # reuse
        buf149 = buf138; del buf138  # reuse
        buf150 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_39], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf147, buf148, buf149, buf150, 9408, 128, grid=grid(9408), stream=stream0)
        buf151 = buf141; del buf141  # reuse
        buf152 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_39], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf148, buf149, buf150, buf151, buf152, 1568, 6, grid=grid(1568), stream=stream0)
        buf154 = reinterpret_tensor(buf146, (8, 768, 196), (150528, 196, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf147, buf151, buf152, arg87_1, arg88_1, buf154, 1204224, grid=grid(1204224), stream=stream0)
        del arg87_1
        del arg88_1
        buf155 = reinterpret_tensor(buf135, (6144, 384), (384, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (6144, 196), (196, 1), 0), reinterpret_tensor(arg89_1, (196, 384), (1, 196), 0), out=buf155)
        del arg89_1
        buf156 = reinterpret_tensor(buf155, (8, 768, 384), (294912, 384, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_236, x_237], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf156, arg90_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg90_1
        buf157 = reinterpret_tensor(buf154, (6144, 196), (196, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (6144, 384), (384, 1), 0), reinterpret_tensor(arg91_1, (384, 196), (1, 384), 0), out=buf157)
        del arg91_1
        buf158 = buf150; del buf150  # reuse
        buf159 = buf149; del buf149  # reuse
        buf160 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_241, layer_norm_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf147, buf157, arg92_1, buf158, buf159, buf160, 9408, 128, grid=grid(9408), stream=stream0)
        buf161 = buf152; del buf152  # reuse
        buf162 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [x_241, layer_norm_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf158, buf159, buf160, buf161, buf162, 1568, 6, grid=grid(1568), stream=stream0)
        buf164 = reinterpret_tensor(buf136, (8, 196, 768), (150528, 768, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_241, layer_norm_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf147, buf157, arg92_1, buf161, buf162, arg93_1, arg94_1, buf164, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg93_1
        del arg94_1
        buf165 = reinterpret_tensor(buf145, (1568, 3072), (3072, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (1568, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 3072), (1, 768), 0), out=buf165)
        del arg95_1
        buf166 = reinterpret_tensor(buf165, (8, 196, 3072), (602112, 3072, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_243], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf166, arg96_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg96_1
        buf167 = reinterpret_tensor(buf164, (1568, 768), (768, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg97_1, (3072, 768), (1, 3072), 0), out=buf167)
        del arg97_1
        buf168 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_247], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf168, buf157, arg92_1, buf167, arg98_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg92_1
        del arg98_1
        buf169 = buf160; del buf160  # reuse
        buf170 = buf159; del buf159  # reuse
        buf171 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_41], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf168, buf169, buf170, buf171, 9408, 128, grid=grid(9408), stream=stream0)
        buf172 = buf162; del buf162  # reuse
        buf173 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_41], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf169, buf170, buf171, buf172, buf173, 1568, 6, grid=grid(1568), stream=stream0)
        buf175 = reinterpret_tensor(buf167, (8, 768, 196), (150528, 196, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_248], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf168, buf172, buf173, arg99_1, arg100_1, buf175, 1204224, grid=grid(1204224), stream=stream0)
        del arg100_1
        del arg99_1
        buf176 = reinterpret_tensor(buf156, (6144, 384), (384, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [x_248], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (6144, 196), (196, 1), 0), reinterpret_tensor(arg101_1, (196, 384), (1, 196), 0), out=buf176)
        del arg101_1
        buf177 = reinterpret_tensor(buf176, (8, 768, 384), (294912, 384, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [x_248, x_249], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf177, arg102_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg102_1
        buf178 = reinterpret_tensor(buf175, (6144, 196), (196, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf177, (6144, 384), (384, 1), 0), reinterpret_tensor(arg103_1, (384, 196), (1, 384), 0), out=buf178)
        del arg103_1
        buf179 = buf171; del buf171  # reuse
        buf180 = buf170; del buf170  # reuse
        buf181 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_253, layer_norm_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf168, buf178, arg104_1, buf179, buf180, buf181, 9408, 128, grid=grid(9408), stream=stream0)
        buf182 = buf173; del buf173  # reuse
        buf183 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [x_253, layer_norm_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf179, buf180, buf181, buf182, buf183, 1568, 6, grid=grid(1568), stream=stream0)
        buf185 = reinterpret_tensor(buf157, (8, 196, 768), (150528, 768, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_253, layer_norm_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf168, buf178, arg104_1, buf182, buf183, arg105_1, arg106_1, buf185, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg105_1
        del arg106_1
        buf186 = reinterpret_tensor(buf166, (1568, 3072), (3072, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf185, (1568, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 3072), (1, 768), 0), out=buf186)
        del arg107_1
        buf187 = reinterpret_tensor(buf186, (8, 196, 3072), (602112, 3072, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_255], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf187, arg108_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg108_1
        buf188 = reinterpret_tensor(buf185, (1568, 768), (768, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg109_1, (3072, 768), (1, 3072), 0), out=buf188)
        del arg109_1
        buf189 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_253, x_259], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf189, buf178, arg104_1, buf188, arg110_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg104_1
        del arg110_1
        buf190 = buf181; del buf181  # reuse
        buf191 = buf180; del buf180  # reuse
        buf192 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_43], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf189, buf190, buf191, buf192, 9408, 128, grid=grid(9408), stream=stream0)
        buf193 = buf183; del buf183  # reuse
        buf194 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_43], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf190, buf191, buf192, buf193, buf194, 1568, 6, grid=grid(1568), stream=stream0)
        buf196 = reinterpret_tensor(buf188, (8, 768, 196), (150528, 196, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_260], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf189, buf193, buf194, arg111_1, arg112_1, buf196, 1204224, grid=grid(1204224), stream=stream0)
        del arg111_1
        del arg112_1
        buf197 = reinterpret_tensor(buf177, (6144, 384), (384, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_260], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (6144, 196), (196, 1), 0), reinterpret_tensor(arg113_1, (196, 384), (1, 196), 0), out=buf197)
        del arg113_1
        buf198 = reinterpret_tensor(buf197, (8, 768, 384), (294912, 384, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [x_260, x_261], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf198, arg114_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg114_1
        buf199 = reinterpret_tensor(buf196, (6144, 196), (196, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (6144, 384), (384, 1), 0), reinterpret_tensor(arg115_1, (384, 196), (1, 384), 0), out=buf199)
        del arg115_1
        buf200 = buf192; del buf192  # reuse
        buf201 = buf191; del buf191  # reuse
        buf202 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_265, layer_norm_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf189, buf199, arg116_1, buf200, buf201, buf202, 9408, 128, grid=grid(9408), stream=stream0)
        buf203 = buf194; del buf194  # reuse
        buf204 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [x_265, layer_norm_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf200, buf201, buf202, buf203, buf204, 1568, 6, grid=grid(1568), stream=stream0)
        buf206 = reinterpret_tensor(buf178, (8, 196, 768), (150528, 768, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_265, layer_norm_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf189, buf199, arg116_1, buf203, buf204, arg117_1, arg118_1, buf206, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg117_1
        del arg118_1
        buf207 = reinterpret_tensor(buf187, (1568, 3072), (3072, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (1568, 768), (768, 1), 0), reinterpret_tensor(arg119_1, (768, 3072), (1, 768), 0), out=buf207)
        del arg119_1
        buf208 = reinterpret_tensor(buf207, (8, 196, 3072), (602112, 3072, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_267], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf208, arg120_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg120_1
        buf209 = reinterpret_tensor(buf206, (1568, 768), (768, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf208, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg121_1, (3072, 768), (1, 3072), 0), out=buf209)
        del arg121_1
        buf210 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_265, x_271], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf210, buf199, arg116_1, buf209, arg122_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg116_1
        del arg122_1
        buf211 = buf202; del buf202  # reuse
        buf212 = buf201; del buf201  # reuse
        buf213 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_45], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf210, buf211, buf212, buf213, 9408, 128, grid=grid(9408), stream=stream0)
        buf214 = buf204; del buf204  # reuse
        buf215 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_45], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf211, buf212, buf213, buf214, buf215, 1568, 6, grid=grid(1568), stream=stream0)
        buf217 = reinterpret_tensor(buf209, (8, 768, 196), (150528, 196, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [x_272], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf210, buf214, buf215, arg123_1, arg124_1, buf217, 1204224, grid=grid(1204224), stream=stream0)
        del arg123_1
        del arg124_1
        buf218 = reinterpret_tensor(buf198, (6144, 384), (384, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [x_272], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (6144, 196), (196, 1), 0), reinterpret_tensor(arg125_1, (196, 384), (1, 196), 0), out=buf218)
        del arg125_1
        buf219 = reinterpret_tensor(buf218, (8, 768, 384), (294912, 384, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [x_272, x_273], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf219, arg126_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg126_1
        buf220 = reinterpret_tensor(buf217, (6144, 196), (196, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (6144, 384), (384, 1), 0), reinterpret_tensor(arg127_1, (384, 196), (1, 384), 0), out=buf220)
        del arg127_1
        buf221 = buf213; del buf213  # reuse
        buf222 = buf212; del buf212  # reuse
        buf223 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [x_277, layer_norm_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf210, buf220, arg128_1, buf221, buf222, buf223, 9408, 128, grid=grid(9408), stream=stream0)
        buf224 = buf215; del buf215  # reuse
        buf225 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [x_277, layer_norm_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf221, buf222, buf223, buf224, buf225, 1568, 6, grid=grid(1568), stream=stream0)
        buf227 = reinterpret_tensor(buf199, (8, 196, 768), (150528, 768, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [x_277, layer_norm_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf210, buf220, arg128_1, buf224, buf225, arg129_1, arg130_1, buf227, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg129_1
        del arg130_1
        buf228 = reinterpret_tensor(buf208, (1568, 3072), (3072, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf227, (1568, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 3072), (1, 768), 0), out=buf228)
        del arg131_1
        buf229 = reinterpret_tensor(buf228, (8, 196, 3072), (602112, 3072, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [x_279], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf229, arg132_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg132_1
        buf230 = reinterpret_tensor(buf227, (1568, 768), (768, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg133_1, (3072, 768), (1, 3072), 0), out=buf230)
        del arg133_1
        buf231 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [x_277, x_283], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf231, buf220, arg128_1, buf230, arg134_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg128_1
        del arg134_1
        buf232 = buf223; del buf223  # reuse
        buf233 = buf222; del buf222  # reuse
        buf234 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_47], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf231, buf232, buf233, buf234, 9408, 128, grid=grid(9408), stream=stream0)
        buf235 = buf225; del buf225  # reuse
        buf236 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_47], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf232, buf233, buf234, buf235, buf236, 1568, 6, grid=grid(1568), stream=stream0)
        buf238 = reinterpret_tensor(buf230, (8, 768, 196), (150528, 196, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [x_284], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf231, buf235, buf236, arg135_1, arg136_1, buf238, 1204224, grid=grid(1204224), stream=stream0)
        del arg135_1
        del arg136_1
        buf239 = reinterpret_tensor(buf219, (6144, 384), (384, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_284], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (6144, 196), (196, 1), 0), reinterpret_tensor(arg137_1, (196, 384), (1, 196), 0), out=buf239)
        del arg137_1
        buf240 = reinterpret_tensor(buf239, (8, 768, 384), (294912, 384, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [x_284, x_285], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf240, arg138_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg138_1
        buf241 = reinterpret_tensor(buf238, (6144, 196), (196, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (6144, 384), (384, 1), 0), reinterpret_tensor(arg139_1, (384, 196), (1, 384), 0), out=buf241)
        del arg139_1
        del buf240
        buf242 = buf234; del buf234  # reuse
        buf243 = buf233; del buf233  # reuse
        buf244 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_289, layer_norm_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf231, buf241, arg140_1, buf242, buf243, buf244, 9408, 128, grid=grid(9408), stream=stream0)
        buf245 = buf236; del buf236  # reuse
        buf246 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [x_289, layer_norm_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf242, buf243, buf244, buf245, buf246, 1568, 6, grid=grid(1568), stream=stream0)
        buf248 = reinterpret_tensor(buf220, (8, 196, 768), (150528, 768, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [x_289, layer_norm_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf231, buf241, arg140_1, buf245, buf246, arg141_1, arg142_1, buf248, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg141_1
        del arg142_1
        buf249 = reinterpret_tensor(buf229, (1568, 3072), (3072, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (1568, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 3072), (1, 768), 0), out=buf249)
        del arg143_1
        buf250 = reinterpret_tensor(buf249, (8, 196, 3072), (602112, 3072, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf250, arg144_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg144_1
        buf251 = reinterpret_tensor(buf248, (1568, 768), (768, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg145_1, (3072, 768), (1, 3072), 0), out=buf251)
        del arg145_1
        del buf250
        buf252 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [x_289, x_295, x_296], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_12.run(buf252, buf241, arg140_1, buf251, arg146_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg140_1
        del arg146_1
        del buf241
        del buf251
        buf253 = buf244; del buf244  # reuse
        buf254 = buf243; del buf243  # reuse
        buf255 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [x_296], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf252, buf253, buf254, buf255, 9408, 128, grid=grid(9408), stream=stream0)
        buf256 = buf246; del buf246  # reuse
        buf257 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [x_296], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf253, buf254, buf255, buf256, buf257, 1568, 6, grid=grid(1568), stream=stream0)
        del buf253
        del buf254
        del buf255
        buf260 = empty_strided_cuda((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_296, x_297], Original ATen: [aten.native_layer_norm, aten.mean]
        triton_per_fused_mean_native_layer_norm_13.run(buf252, buf256, buf257, arg147_1, arg148_1, buf260, 6144, 196, grid=grid(6144), stream=stream0)
        del arg147_1
        del arg148_1
        del buf252
        del buf256
        del buf257
        buf261 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_296, x_297, x_299], Original ATen: [aten.native_layer_norm, aten.mean, aten.addmm]
        extern_kernels.addmm(arg150_1, buf260, reinterpret_tensor(arg149_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf261)
        del arg149_1
        del arg150_1
        del buf260
    return (buf261, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mixer_b16_224', benchmark_compiled_module)
