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


# kernel path: /tmp/torchinductor_sahanp/y5/cy5ainy2ssmnjlx6dcj6hc7iogepzhspameywpyxmipddbbma5qc.py
# Topologically Sorted Source Nodes: [layer_norm_49], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_49 => clone_170, var_mean_49
# Graph fragment:
#   %clone_170 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_146,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_49 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_170, [2]), kwargs = {correction: 0, keepdim: True})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 3
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


# kernel path: /tmp/torchinductor_sahanp/af/cafd62sziai2q7vqvml2xyems3us4wv2jf2yh5t3cse22ift7u4c.py
# Topologically Sorted Source Nodes: [layer_norm_49], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_49 => clone_170, var_mean_49
# Graph fragment:
#   %clone_170 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_146,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_49 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_170, [2]), kwargs = {correction: 0, keepdim: True})
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
    reduction_hint=ReductionHint.OUTER,
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
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/ej/cejuoyhkusqw7xzhmgiwdzyy3ow6saaeb3qo7texrk2kxace6yv3.py
# Topologically Sorted Source Nodes: [x_296], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_296 => clone_171
# Graph fragment:
#   %clone_171 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_147,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 196) % 384
    x0 = xindex % 196
    x2 = (xindex // 75264)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 384.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3f/c3fdubgnxjos2ztgtvmtnzej4du7enrko6ud532amvk7ushkbclm.py
# Topologically Sorted Source Nodes: [silu_48, x_297], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   silu_48 => mul_196, sigmoid_48
#   x_297 => mul_197
# Graph fragment:
#   %sigmoid_48 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%getitem_197,), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_197, %sigmoid_48), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_196, %mul_196), kwargs = {})
triton_poi_fused_mul_silu_3 = async_compile.triton('triton_poi_fused_mul_silu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 192
    x1 = (xindex // 192)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (192 + x0 + (384*x1)), None)
    tmp4 = tl.load(in_ptr1 + (192 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h2/ch2o5kzzvsjwuskye7snqt6wgj73wtq6zgq46eszztf34usbdyeb.py
# Topologically Sorted Source Nodes: [x_301, layer_norm_50], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_50 => clone_174, var_mean_50
#   x_301 => add_173
# Graph fragment:
#   %add_173 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_146, %permute_150), kwargs = {})
#   %clone_174 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_173,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_50 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_174, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_4 = async_compile.triton('triton_red_fused_add_native_layer_norm_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 3
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


# kernel path: /tmp/torchinductor_sahanp/56/c56bql7pshh54zidmzlogzzmvkw7khh74yevmy5x2j35r5tzth5h.py
# Topologically Sorted Source Nodes: [x_301, layer_norm_50], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_50 => add_174, add_175, clone_174, mul_198, mul_199, rsqrt_50, sub_50, var_mean_50
#   x_301 => add_173
# Graph fragment:
#   %add_173 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_146, %permute_150), kwargs = {})
#   %clone_174 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_173,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_50 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_174, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_174, %getitem_199), kwargs = {})
#   %add_174 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_198, 1e-06), kwargs = {})
#   %rsqrt_50 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_174,), kwargs = {})
#   %mul_198 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %rsqrt_50), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_198, %arg9_1), kwargs = {})
#   %add_175 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_199, %arg10_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_5 = async_compile.triton('triton_poi_fused_add_native_layer_norm_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
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
    tmp10 = 384.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kh/ckhagxmlic73u6kpk3y6fcokagxi5y3x7uhgk623d3h6dyisx57k.py
# Topologically Sorted Source Nodes: [silu_49, x_303], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   silu_49 => mul_200, sigmoid_49
#   x_303 => mul_201
# Graph fragment:
#   %sigmoid_49 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%getitem_201,), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_201, %sigmoid_49), kwargs = {})
#   %mul_201 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_200, %mul_200), kwargs = {})
triton_poi_fused_mul_silu_6 = async_compile.triton('triton_poi_fused_mul_silu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (768 + x0 + (1536*x1)), None)
    tmp4 = tl.load(in_ptr1 + (768 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n3/cn3rfsuoz3zl4csjcobrn5s56ppwetkh35merladz6cih47og6a6.py
# Topologically Sorted Source Nodes: [x_301, x_307], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_301 => add_173
#   x_307 => add_176
# Graph fragment:
#   %add_173 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_146, %permute_150), kwargs = {})
#   %add_176 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_173, %view_201), kwargs = {})
triton_poi_fused_add_7 = async_compile.triton('triton_poi_fused_add_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4s/c4sd4vtggvwexbe7y32bazarp4tsachiq6gcinc2w53hhpnpys4g.py
# Topologically Sorted Source Nodes: [layer_norm_51], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_51 => clone_177, var_mean_51
# Graph fragment:
#   %clone_177 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_176,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_51 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_177, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_layer_norm_8 = async_compile.triton('triton_red_fused_native_layer_norm_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_8(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
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


# kernel path: /tmp/torchinductor_sahanp/sk/csk2bqb4zfeeqbbe2d5ukavfyobqtatlz2e4wenvmof7fpyy5zxe.py
# Topologically Sorted Source Nodes: [x_308], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_308 => clone_178
# Graph fragment:
#   %clone_178 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_153,), kwargs = {memory_format: torch.contiguous_format})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 196
    x2 = (xindex // 75264)
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 384.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/op/coph2u4buoonta3ewlz3b27irjjwokon3f6tcelj5a45rduawokz.py
# Topologically Sorted Source Nodes: [x_313, layer_norm_52], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_52 => clone_181, var_mean_52
#   x_313 => add_180
# Graph fragment:
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_176, %permute_156), kwargs = {})
#   %clone_181 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_180,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_52 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_181, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_10 = async_compile.triton('triton_red_fused_add_native_layer_norm_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
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


# kernel path: /tmp/torchinductor_sahanp/qa/cqajujyd4ya3idghlxsiw2gmrrckczpyecbdikkefyewn2djptyw.py
# Topologically Sorted Source Nodes: [x_313, layer_norm_52], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_52 => add_181, add_182, clone_181, mul_206, mul_207, rsqrt_52, sub_52, var_mean_52
#   x_313 => add_180
# Graph fragment:
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_176, %permute_156), kwargs = {})
#   %clone_181 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_180,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_52 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_181, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_181, %getitem_207), kwargs = {})
#   %add_181 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_206, 1e-06), kwargs = {})
#   %rsqrt_52 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_181,), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %rsqrt_52), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_206, %arg21_1), kwargs = {})
#   %add_182 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_207, %arg22_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_11 = async_compile.triton('triton_poi_fused_add_native_layer_norm_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 384)
    y0 = yindex % 384
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
    tmp8 = 384.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ta/ctatsyh2frlflgri74lohcf6jlcw6y5d5z3do7epu7erblwokaqa.py
# Topologically Sorted Source Nodes: [x_313, x_319], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_313 => add_180
#   x_319 => add_183
# Graph fragment:
#   %add_180 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_176, %permute_156), kwargs = {})
#   %add_183 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_180, %view_209), kwargs = {})
triton_poi_fused_add_12 = async_compile.triton('triton_poi_fused_add_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/na/cnafsjas2c3fs5piox6ofx2xmwj3223pvoljczbfj4sjcsdi4s6q.py
# Topologically Sorted Source Nodes: [x_584, x_585], Original ATen: [aten.native_layer_norm, aten.mean]
# Source node to ATen node mapping:
#   x_584 => add_338, add_339, mul_386, mul_387, rsqrt_97, sub_97, var_mean_97
#   x_585 => mean_1
# Graph fragment:
#   %var_mean_97 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_338, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_97 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_338, %getitem_387), kwargs = {})
#   %add_338 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_386, 1e-06), kwargs = {})
#   %rsqrt_97 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_338,), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_97, %rsqrt_97), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_386, %arg291_1), kwargs = {})
#   %add_339 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_387, %arg292_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_339, [1]), kwargs = {})
triton_per_fused_mean_native_layer_norm_13 = async_compile.triton('triton_per_fused_mean_native_layer_norm_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_native_layer_norm_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
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
    x1 = (xindex // 384)
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (196*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (196*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 384.0
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg1_1, (384, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg2_1, (384, ), (1, ))
    assert_size_stride(arg3_1, (384, ), (1, ))
    assert_size_stride(arg4_1, (384, ), (1, ))
    assert_size_stride(arg5_1, (384, 196), (196, 1))
    assert_size_stride(arg6_1, (384, ), (1, ))
    assert_size_stride(arg7_1, (196, 192), (192, 1))
    assert_size_stride(arg8_1, (196, ), (1, ))
    assert_size_stride(arg9_1, (384, ), (1, ))
    assert_size_stride(arg10_1, (384, ), (1, ))
    assert_size_stride(arg11_1, (1536, 384), (384, 1))
    assert_size_stride(arg12_1, (1536, ), (1, ))
    assert_size_stride(arg13_1, (384, 768), (768, 1))
    assert_size_stride(arg14_1, (384, ), (1, ))
    assert_size_stride(arg15_1, (384, ), (1, ))
    assert_size_stride(arg16_1, (384, ), (1, ))
    assert_size_stride(arg17_1, (384, 196), (196, 1))
    assert_size_stride(arg18_1, (384, ), (1, ))
    assert_size_stride(arg19_1, (196, 192), (192, 1))
    assert_size_stride(arg20_1, (196, ), (1, ))
    assert_size_stride(arg21_1, (384, ), (1, ))
    assert_size_stride(arg22_1, (384, ), (1, ))
    assert_size_stride(arg23_1, (1536, 384), (384, 1))
    assert_size_stride(arg24_1, (1536, ), (1, ))
    assert_size_stride(arg25_1, (384, 768), (768, 1))
    assert_size_stride(arg26_1, (384, ), (1, ))
    assert_size_stride(arg27_1, (384, ), (1, ))
    assert_size_stride(arg28_1, (384, ), (1, ))
    assert_size_stride(arg29_1, (384, 196), (196, 1))
    assert_size_stride(arg30_1, (384, ), (1, ))
    assert_size_stride(arg31_1, (196, 192), (192, 1))
    assert_size_stride(arg32_1, (196, ), (1, ))
    assert_size_stride(arg33_1, (384, ), (1, ))
    assert_size_stride(arg34_1, (384, ), (1, ))
    assert_size_stride(arg35_1, (1536, 384), (384, 1))
    assert_size_stride(arg36_1, (1536, ), (1, ))
    assert_size_stride(arg37_1, (384, 768), (768, 1))
    assert_size_stride(arg38_1, (384, ), (1, ))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (384, ), (1, ))
    assert_size_stride(arg41_1, (384, 196), (196, 1))
    assert_size_stride(arg42_1, (384, ), (1, ))
    assert_size_stride(arg43_1, (196, 192), (192, 1))
    assert_size_stride(arg44_1, (196, ), (1, ))
    assert_size_stride(arg45_1, (384, ), (1, ))
    assert_size_stride(arg46_1, (384, ), (1, ))
    assert_size_stride(arg47_1, (1536, 384), (384, 1))
    assert_size_stride(arg48_1, (1536, ), (1, ))
    assert_size_stride(arg49_1, (384, 768), (768, 1))
    assert_size_stride(arg50_1, (384, ), (1, ))
    assert_size_stride(arg51_1, (384, ), (1, ))
    assert_size_stride(arg52_1, (384, ), (1, ))
    assert_size_stride(arg53_1, (384, 196), (196, 1))
    assert_size_stride(arg54_1, (384, ), (1, ))
    assert_size_stride(arg55_1, (196, 192), (192, 1))
    assert_size_stride(arg56_1, (196, ), (1, ))
    assert_size_stride(arg57_1, (384, ), (1, ))
    assert_size_stride(arg58_1, (384, ), (1, ))
    assert_size_stride(arg59_1, (1536, 384), (384, 1))
    assert_size_stride(arg60_1, (1536, ), (1, ))
    assert_size_stride(arg61_1, (384, 768), (768, 1))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (384, ), (1, ))
    assert_size_stride(arg64_1, (384, ), (1, ))
    assert_size_stride(arg65_1, (384, 196), (196, 1))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (196, 192), (192, 1))
    assert_size_stride(arg68_1, (196, ), (1, ))
    assert_size_stride(arg69_1, (384, ), (1, ))
    assert_size_stride(arg70_1, (384, ), (1, ))
    assert_size_stride(arg71_1, (1536, 384), (384, 1))
    assert_size_stride(arg72_1, (1536, ), (1, ))
    assert_size_stride(arg73_1, (384, 768), (768, 1))
    assert_size_stride(arg74_1, (384, ), (1, ))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (384, ), (1, ))
    assert_size_stride(arg77_1, (384, 196), (196, 1))
    assert_size_stride(arg78_1, (384, ), (1, ))
    assert_size_stride(arg79_1, (196, 192), (192, 1))
    assert_size_stride(arg80_1, (196, ), (1, ))
    assert_size_stride(arg81_1, (384, ), (1, ))
    assert_size_stride(arg82_1, (384, ), (1, ))
    assert_size_stride(arg83_1, (1536, 384), (384, 1))
    assert_size_stride(arg84_1, (1536, ), (1, ))
    assert_size_stride(arg85_1, (384, 768), (768, 1))
    assert_size_stride(arg86_1, (384, ), (1, ))
    assert_size_stride(arg87_1, (384, ), (1, ))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (384, 196), (196, 1))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (196, 192), (192, 1))
    assert_size_stride(arg92_1, (196, ), (1, ))
    assert_size_stride(arg93_1, (384, ), (1, ))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (1536, 384), (384, 1))
    assert_size_stride(arg96_1, (1536, ), (1, ))
    assert_size_stride(arg97_1, (384, 768), (768, 1))
    assert_size_stride(arg98_1, (384, ), (1, ))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (384, ), (1, ))
    assert_size_stride(arg101_1, (384, 196), (196, 1))
    assert_size_stride(arg102_1, (384, ), (1, ))
    assert_size_stride(arg103_1, (196, 192), (192, 1))
    assert_size_stride(arg104_1, (196, ), (1, ))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (384, ), (1, ))
    assert_size_stride(arg107_1, (1536, 384), (384, 1))
    assert_size_stride(arg108_1, (1536, ), (1, ))
    assert_size_stride(arg109_1, (384, 768), (768, 1))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (384, 196), (196, 1))
    assert_size_stride(arg114_1, (384, ), (1, ))
    assert_size_stride(arg115_1, (196, 192), (192, 1))
    assert_size_stride(arg116_1, (196, ), (1, ))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (384, ), (1, ))
    assert_size_stride(arg119_1, (1536, 384), (384, 1))
    assert_size_stride(arg120_1, (1536, ), (1, ))
    assert_size_stride(arg121_1, (384, 768), (768, 1))
    assert_size_stride(arg122_1, (384, ), (1, ))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (384, ), (1, ))
    assert_size_stride(arg125_1, (384, 196), (196, 1))
    assert_size_stride(arg126_1, (384, ), (1, ))
    assert_size_stride(arg127_1, (196, 192), (192, 1))
    assert_size_stride(arg128_1, (196, ), (1, ))
    assert_size_stride(arg129_1, (384, ), (1, ))
    assert_size_stride(arg130_1, (384, ), (1, ))
    assert_size_stride(arg131_1, (1536, 384), (384, 1))
    assert_size_stride(arg132_1, (1536, ), (1, ))
    assert_size_stride(arg133_1, (384, 768), (768, 1))
    assert_size_stride(arg134_1, (384, ), (1, ))
    assert_size_stride(arg135_1, (384, ), (1, ))
    assert_size_stride(arg136_1, (384, ), (1, ))
    assert_size_stride(arg137_1, (384, 196), (196, 1))
    assert_size_stride(arg138_1, (384, ), (1, ))
    assert_size_stride(arg139_1, (196, 192), (192, 1))
    assert_size_stride(arg140_1, (196, ), (1, ))
    assert_size_stride(arg141_1, (384, ), (1, ))
    assert_size_stride(arg142_1, (384, ), (1, ))
    assert_size_stride(arg143_1, (1536, 384), (384, 1))
    assert_size_stride(arg144_1, (1536, ), (1, ))
    assert_size_stride(arg145_1, (384, 768), (768, 1))
    assert_size_stride(arg146_1, (384, ), (1, ))
    assert_size_stride(arg147_1, (384, ), (1, ))
    assert_size_stride(arg148_1, (384, ), (1, ))
    assert_size_stride(arg149_1, (384, 196), (196, 1))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (196, 192), (192, 1))
    assert_size_stride(arg152_1, (196, ), (1, ))
    assert_size_stride(arg153_1, (384, ), (1, ))
    assert_size_stride(arg154_1, (384, ), (1, ))
    assert_size_stride(arg155_1, (1536, 384), (384, 1))
    assert_size_stride(arg156_1, (1536, ), (1, ))
    assert_size_stride(arg157_1, (384, 768), (768, 1))
    assert_size_stride(arg158_1, (384, ), (1, ))
    assert_size_stride(arg159_1, (384, ), (1, ))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (384, 196), (196, 1))
    assert_size_stride(arg162_1, (384, ), (1, ))
    assert_size_stride(arg163_1, (196, 192), (192, 1))
    assert_size_stride(arg164_1, (196, ), (1, ))
    assert_size_stride(arg165_1, (384, ), (1, ))
    assert_size_stride(arg166_1, (384, ), (1, ))
    assert_size_stride(arg167_1, (1536, 384), (384, 1))
    assert_size_stride(arg168_1, (1536, ), (1, ))
    assert_size_stride(arg169_1, (384, 768), (768, 1))
    assert_size_stride(arg170_1, (384, ), (1, ))
    assert_size_stride(arg171_1, (384, ), (1, ))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (384, 196), (196, 1))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (196, 192), (192, 1))
    assert_size_stride(arg176_1, (196, ), (1, ))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (384, ), (1, ))
    assert_size_stride(arg179_1, (1536, 384), (384, 1))
    assert_size_stride(arg180_1, (1536, ), (1, ))
    assert_size_stride(arg181_1, (384, 768), (768, 1))
    assert_size_stride(arg182_1, (384, ), (1, ))
    assert_size_stride(arg183_1, (384, ), (1, ))
    assert_size_stride(arg184_1, (384, ), (1, ))
    assert_size_stride(arg185_1, (384, 196), (196, 1))
    assert_size_stride(arg186_1, (384, ), (1, ))
    assert_size_stride(arg187_1, (196, 192), (192, 1))
    assert_size_stride(arg188_1, (196, ), (1, ))
    assert_size_stride(arg189_1, (384, ), (1, ))
    assert_size_stride(arg190_1, (384, ), (1, ))
    assert_size_stride(arg191_1, (1536, 384), (384, 1))
    assert_size_stride(arg192_1, (1536, ), (1, ))
    assert_size_stride(arg193_1, (384, 768), (768, 1))
    assert_size_stride(arg194_1, (384, ), (1, ))
    assert_size_stride(arg195_1, (384, ), (1, ))
    assert_size_stride(arg196_1, (384, ), (1, ))
    assert_size_stride(arg197_1, (384, 196), (196, 1))
    assert_size_stride(arg198_1, (384, ), (1, ))
    assert_size_stride(arg199_1, (196, 192), (192, 1))
    assert_size_stride(arg200_1, (196, ), (1, ))
    assert_size_stride(arg201_1, (384, ), (1, ))
    assert_size_stride(arg202_1, (384, ), (1, ))
    assert_size_stride(arg203_1, (1536, 384), (384, 1))
    assert_size_stride(arg204_1, (1536, ), (1, ))
    assert_size_stride(arg205_1, (384, 768), (768, 1))
    assert_size_stride(arg206_1, (384, ), (1, ))
    assert_size_stride(arg207_1, (384, ), (1, ))
    assert_size_stride(arg208_1, (384, ), (1, ))
    assert_size_stride(arg209_1, (384, 196), (196, 1))
    assert_size_stride(arg210_1, (384, ), (1, ))
    assert_size_stride(arg211_1, (196, 192), (192, 1))
    assert_size_stride(arg212_1, (196, ), (1, ))
    assert_size_stride(arg213_1, (384, ), (1, ))
    assert_size_stride(arg214_1, (384, ), (1, ))
    assert_size_stride(arg215_1, (1536, 384), (384, 1))
    assert_size_stride(arg216_1, (1536, ), (1, ))
    assert_size_stride(arg217_1, (384, 768), (768, 1))
    assert_size_stride(arg218_1, (384, ), (1, ))
    assert_size_stride(arg219_1, (384, ), (1, ))
    assert_size_stride(arg220_1, (384, ), (1, ))
    assert_size_stride(arg221_1, (384, 196), (196, 1))
    assert_size_stride(arg222_1, (384, ), (1, ))
    assert_size_stride(arg223_1, (196, 192), (192, 1))
    assert_size_stride(arg224_1, (196, ), (1, ))
    assert_size_stride(arg225_1, (384, ), (1, ))
    assert_size_stride(arg226_1, (384, ), (1, ))
    assert_size_stride(arg227_1, (1536, 384), (384, 1))
    assert_size_stride(arg228_1, (1536, ), (1, ))
    assert_size_stride(arg229_1, (384, 768), (768, 1))
    assert_size_stride(arg230_1, (384, ), (1, ))
    assert_size_stride(arg231_1, (384, ), (1, ))
    assert_size_stride(arg232_1, (384, ), (1, ))
    assert_size_stride(arg233_1, (384, 196), (196, 1))
    assert_size_stride(arg234_1, (384, ), (1, ))
    assert_size_stride(arg235_1, (196, 192), (192, 1))
    assert_size_stride(arg236_1, (196, ), (1, ))
    assert_size_stride(arg237_1, (384, ), (1, ))
    assert_size_stride(arg238_1, (384, ), (1, ))
    assert_size_stride(arg239_1, (1536, 384), (384, 1))
    assert_size_stride(arg240_1, (1536, ), (1, ))
    assert_size_stride(arg241_1, (384, 768), (768, 1))
    assert_size_stride(arg242_1, (384, ), (1, ))
    assert_size_stride(arg243_1, (384, ), (1, ))
    assert_size_stride(arg244_1, (384, ), (1, ))
    assert_size_stride(arg245_1, (384, 196), (196, 1))
    assert_size_stride(arg246_1, (384, ), (1, ))
    assert_size_stride(arg247_1, (196, 192), (192, 1))
    assert_size_stride(arg248_1, (196, ), (1, ))
    assert_size_stride(arg249_1, (384, ), (1, ))
    assert_size_stride(arg250_1, (384, ), (1, ))
    assert_size_stride(arg251_1, (1536, 384), (384, 1))
    assert_size_stride(arg252_1, (1536, ), (1, ))
    assert_size_stride(arg253_1, (384, 768), (768, 1))
    assert_size_stride(arg254_1, (384, ), (1, ))
    assert_size_stride(arg255_1, (384, ), (1, ))
    assert_size_stride(arg256_1, (384, ), (1, ))
    assert_size_stride(arg257_1, (384, 196), (196, 1))
    assert_size_stride(arg258_1, (384, ), (1, ))
    assert_size_stride(arg259_1, (196, 192), (192, 1))
    assert_size_stride(arg260_1, (196, ), (1, ))
    assert_size_stride(arg261_1, (384, ), (1, ))
    assert_size_stride(arg262_1, (384, ), (1, ))
    assert_size_stride(arg263_1, (1536, 384), (384, 1))
    assert_size_stride(arg264_1, (1536, ), (1, ))
    assert_size_stride(arg265_1, (384, 768), (768, 1))
    assert_size_stride(arg266_1, (384, ), (1, ))
    assert_size_stride(arg267_1, (384, ), (1, ))
    assert_size_stride(arg268_1, (384, ), (1, ))
    assert_size_stride(arg269_1, (384, 196), (196, 1))
    assert_size_stride(arg270_1, (384, ), (1, ))
    assert_size_stride(arg271_1, (196, 192), (192, 1))
    assert_size_stride(arg272_1, (196, ), (1, ))
    assert_size_stride(arg273_1, (384, ), (1, ))
    assert_size_stride(arg274_1, (384, ), (1, ))
    assert_size_stride(arg275_1, (1536, 384), (384, 1))
    assert_size_stride(arg276_1, (1536, ), (1, ))
    assert_size_stride(arg277_1, (384, 768), (768, 1))
    assert_size_stride(arg278_1, (384, ), (1, ))
    assert_size_stride(arg279_1, (384, ), (1, ))
    assert_size_stride(arg280_1, (384, ), (1, ))
    assert_size_stride(arg281_1, (384, 196), (196, 1))
    assert_size_stride(arg282_1, (384, ), (1, ))
    assert_size_stride(arg283_1, (196, 192), (192, 1))
    assert_size_stride(arg284_1, (196, ), (1, ))
    assert_size_stride(arg285_1, (384, ), (1, ))
    assert_size_stride(arg286_1, (384, ), (1, ))
    assert_size_stride(arg287_1, (1536, 384), (384, 1))
    assert_size_stride(arg288_1, (1536, ), (1, ))
    assert_size_stride(arg289_1, (384, 768), (768, 1))
    assert_size_stride(arg290_1, (384, ), (1, ))
    assert_size_stride(arg291_1, (384, ), (1, ))
    assert_size_stride(arg292_1, (384, ), (1, ))
    assert_size_stride(arg293_1, (1000, 384), (384, 1))
    assert_size_stride(arg294_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x_294], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((8, 196, 1, 3), (588, 1, 4704, 196), torch.float32)
        buf2 = empty_strided_cuda((8, 196, 1, 3), (588, 1, 4704, 196), torch.float32)
        buf3 = empty_strided_cuda((8, 196, 1, 3), (588, 1, 4704, 196), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_49], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf0, arg2_1, buf1, buf2, buf3, 4704, 128, grid=grid(4704), stream=stream0)
        buf4 = empty_strided_cuda((8, 196, 1), (196, 1, 1568), torch.float32)
        buf5 = empty_strided_cuda((8, 196, 1), (196, 1, 1568), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_49], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 1568, 3, grid=grid(1568), stream=stream0)
        buf7 = empty_strided_cuda((8, 384, 196), (75264, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_296], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf0, arg2_1, buf4, buf5, arg3_1, arg4_1, buf7, 602112, grid=grid(602112), stream=stream0)
        del arg3_1
        del arg4_1
        buf8 = empty_strided_cuda((3072, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_296], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (3072, 196), (196, 1), 0), reinterpret_tensor(arg5_1, (196, 384), (1, 196), 0), out=buf8)
        del arg5_1
        buf9 = empty_strided_cuda((8, 384, 192), (73728, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [silu_48, x_297], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf8, arg6_1, buf9, 589824, grid=grid(589824), stream=stream0)
        del arg6_1
        buf10 = reinterpret_tensor(buf7, (3072, 196), (196, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf9, (3072, 192), (192, 1), 0), reinterpret_tensor(arg7_1, (192, 196), (1, 192), 0), out=buf10)
        del arg7_1
        buf11 = buf3; del buf3  # reuse
        buf12 = buf2; del buf2  # reuse
        buf13 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_301, layer_norm_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf0, arg2_1, buf10, arg8_1, buf11, buf12, buf13, 4704, 128, grid=grid(4704), stream=stream0)
        buf14 = buf5; del buf5  # reuse
        buf15 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_301, layer_norm_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf11, buf12, buf13, buf14, buf15, 1568, 3, grid=grid(1568), stream=stream0)
        buf17 = empty_strided_cuda((8, 196, 384), (75264, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_301, layer_norm_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_5.run(buf0, arg2_1, buf10, arg8_1, buf14, buf15, arg9_1, arg10_1, buf17, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg10_1
        del arg9_1
        buf18 = empty_strided_cuda((1568, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf17, (1568, 384), (384, 1), 0), reinterpret_tensor(arg11_1, (384, 1536), (1, 384), 0), out=buf18)
        del arg11_1
        buf19 = empty_strided_cuda((8, 196, 768), (150528, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [silu_49, x_303], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf18, arg12_1, buf19, 1204224, grid=grid(1204224), stream=stream0)
        del arg12_1
        buf20 = reinterpret_tensor(buf17, (1568, 384), (384, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (1568, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 384), (1, 768), 0), out=buf20)
        del arg13_1
        buf21 = reinterpret_tensor(buf0, (8, 196, 384), (75264, 1, 196), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_301, x_307], Original ATen: [aten.add]
        triton_poi_fused_add_7.run(buf21, arg2_1, buf10, arg8_1, buf20, arg14_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg14_1
        del arg2_1
        del arg8_1
        buf22 = buf13; del buf13  # reuse
        buf23 = buf12; del buf12  # reuse
        buf24 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_51], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf21, buf22, buf23, buf24, 4704, 128, grid=grid(4704), stream=stream0)
        buf25 = buf15; del buf15  # reuse
        buf26 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_51], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf22, buf23, buf24, buf25, buf26, 1568, 3, grid=grid(1568), stream=stream0)
        buf28 = reinterpret_tensor(buf20, (8, 384, 196), (75264, 196, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_308], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf21, buf25, buf26, arg15_1, arg16_1, buf28, 602112, grid=grid(602112), stream=stream0)
        del arg15_1
        del arg16_1
        buf29 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_308], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (3072, 196), (196, 1), 0), reinterpret_tensor(arg17_1, (196, 384), (1, 196), 0), out=buf29)
        del arg17_1
        buf30 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [silu_50, x_309], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf29, arg18_1, buf30, 589824, grid=grid(589824), stream=stream0)
        del arg18_1
        buf31 = reinterpret_tensor(buf28, (3072, 196), (196, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (3072, 192), (192, 1), 0), reinterpret_tensor(arg19_1, (192, 196), (1, 192), 0), out=buf31)
        del arg19_1
        buf32 = buf24; del buf24  # reuse
        buf33 = buf23; del buf23  # reuse
        buf34 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_313, layer_norm_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf21, buf31, arg20_1, buf32, buf33, buf34, 4704, 128, grid=grid(4704), stream=stream0)
        buf35 = buf26; del buf26  # reuse
        buf36 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_313, layer_norm_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf32, buf33, buf34, buf35, buf36, 1568, 3, grid=grid(1568), stream=stream0)
        buf38 = reinterpret_tensor(buf10, (8, 196, 384), (75264, 384, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_313, layer_norm_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf21, buf31, arg20_1, buf35, buf36, arg21_1, arg22_1, buf38, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg21_1
        del arg22_1
        buf39 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (1568, 384), (384, 1), 0), reinterpret_tensor(arg23_1, (384, 1536), (1, 384), 0), out=buf39)
        del arg23_1
        buf40 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [silu_51, x_315], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf39, arg24_1, buf40, 1204224, grid=grid(1204224), stream=stream0)
        del arg24_1
        buf41 = reinterpret_tensor(buf38, (1568, 384), (384, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (1568, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 384), (1, 768), 0), out=buf41)
        del arg25_1
        buf42 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_313, x_319], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf42, buf31, arg20_1, buf41, arg26_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg20_1
        del arg26_1
        buf43 = buf34; del buf34  # reuse
        buf44 = buf33; del buf33  # reuse
        buf45 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_53], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf42, buf43, buf44, buf45, 4704, 128, grid=grid(4704), stream=stream0)
        buf46 = buf36; del buf36  # reuse
        buf47 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_53], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf43, buf44, buf45, buf46, buf47, 1568, 3, grid=grid(1568), stream=stream0)
        buf49 = reinterpret_tensor(buf41, (8, 384, 196), (75264, 196, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_320], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf42, buf46, buf47, arg27_1, arg28_1, buf49, 602112, grid=grid(602112), stream=stream0)
        del arg27_1
        del arg28_1
        buf50 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_320], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (3072, 196), (196, 1), 0), reinterpret_tensor(arg29_1, (196, 384), (1, 196), 0), out=buf50)
        del arg29_1
        buf51 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [silu_52, x_321], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf50, arg30_1, buf51, 589824, grid=grid(589824), stream=stream0)
        del arg30_1
        buf52 = reinterpret_tensor(buf49, (3072, 196), (196, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (3072, 192), (192, 1), 0), reinterpret_tensor(arg31_1, (192, 196), (1, 192), 0), out=buf52)
        del arg31_1
        buf53 = buf45; del buf45  # reuse
        buf54 = buf44; del buf44  # reuse
        buf55 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_325, layer_norm_54], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf42, buf52, arg32_1, buf53, buf54, buf55, 4704, 128, grid=grid(4704), stream=stream0)
        buf56 = buf47; del buf47  # reuse
        buf57 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_325, layer_norm_54], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf53, buf54, buf55, buf56, buf57, 1568, 3, grid=grid(1568), stream=stream0)
        buf59 = reinterpret_tensor(buf31, (8, 196, 384), (75264, 384, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_325, layer_norm_54], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf42, buf52, arg32_1, buf56, buf57, arg33_1, arg34_1, buf59, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg33_1
        del arg34_1
        buf60 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (1568, 384), (384, 1), 0), reinterpret_tensor(arg35_1, (384, 1536), (1, 384), 0), out=buf60)
        del arg35_1
        buf61 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [silu_53, x_327], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf60, arg36_1, buf61, 1204224, grid=grid(1204224), stream=stream0)
        del arg36_1
        buf62 = reinterpret_tensor(buf59, (1568, 384), (384, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (1568, 768), (768, 1), 0), reinterpret_tensor(arg37_1, (768, 384), (1, 768), 0), out=buf62)
        del arg37_1
        buf63 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_325, x_331], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf63, buf52, arg32_1, buf62, arg38_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg32_1
        del arg38_1
        buf64 = buf55; del buf55  # reuse
        buf65 = buf54; del buf54  # reuse
        buf66 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_55], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf63, buf64, buf65, buf66, 4704, 128, grid=grid(4704), stream=stream0)
        buf67 = buf57; del buf57  # reuse
        buf68 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_55], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf64, buf65, buf66, buf67, buf68, 1568, 3, grid=grid(1568), stream=stream0)
        buf70 = reinterpret_tensor(buf62, (8, 384, 196), (75264, 196, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_332], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf63, buf67, buf68, arg39_1, arg40_1, buf70, 602112, grid=grid(602112), stream=stream0)
        del arg39_1
        del arg40_1
        buf71 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_332], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (3072, 196), (196, 1), 0), reinterpret_tensor(arg41_1, (196, 384), (1, 196), 0), out=buf71)
        del arg41_1
        buf72 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [silu_54, x_333], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf71, arg42_1, buf72, 589824, grid=grid(589824), stream=stream0)
        del arg42_1
        buf73 = reinterpret_tensor(buf70, (3072, 196), (196, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (3072, 192), (192, 1), 0), reinterpret_tensor(arg43_1, (192, 196), (1, 192), 0), out=buf73)
        del arg43_1
        buf74 = buf66; del buf66  # reuse
        buf75 = buf65; del buf65  # reuse
        buf76 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_337, layer_norm_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf63, buf73, arg44_1, buf74, buf75, buf76, 4704, 128, grid=grid(4704), stream=stream0)
        buf77 = buf68; del buf68  # reuse
        buf78 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_337, layer_norm_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf74, buf75, buf76, buf77, buf78, 1568, 3, grid=grid(1568), stream=stream0)
        buf80 = reinterpret_tensor(buf52, (8, 196, 384), (75264, 384, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_337, layer_norm_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf63, buf73, arg44_1, buf77, buf78, arg45_1, arg46_1, buf80, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg45_1
        del arg46_1
        buf81 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (1568, 384), (384, 1), 0), reinterpret_tensor(arg47_1, (384, 1536), (1, 384), 0), out=buf81)
        del arg47_1
        buf82 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [silu_55, x_339], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf81, arg48_1, buf82, 1204224, grid=grid(1204224), stream=stream0)
        del arg48_1
        buf83 = reinterpret_tensor(buf80, (1568, 384), (384, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (1568, 768), (768, 1), 0), reinterpret_tensor(arg49_1, (768, 384), (1, 768), 0), out=buf83)
        del arg49_1
        buf84 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [x_337, x_343], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf84, buf73, arg44_1, buf83, arg50_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg44_1
        del arg50_1
        buf85 = buf76; del buf76  # reuse
        buf86 = buf75; del buf75  # reuse
        buf87 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_57], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf84, buf85, buf86, buf87, 4704, 128, grid=grid(4704), stream=stream0)
        buf88 = buf78; del buf78  # reuse
        buf89 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_57], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf85, buf86, buf87, buf88, buf89, 1568, 3, grid=grid(1568), stream=stream0)
        buf91 = reinterpret_tensor(buf83, (8, 384, 196), (75264, 196, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_344], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf84, buf88, buf89, arg51_1, arg52_1, buf91, 602112, grid=grid(602112), stream=stream0)
        del arg51_1
        del arg52_1
        buf92 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_344], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (3072, 196), (196, 1), 0), reinterpret_tensor(arg53_1, (196, 384), (1, 196), 0), out=buf92)
        del arg53_1
        buf93 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [silu_56, x_345], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf92, arg54_1, buf93, 589824, grid=grid(589824), stream=stream0)
        del arg54_1
        buf94 = reinterpret_tensor(buf91, (3072, 196), (196, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (3072, 192), (192, 1), 0), reinterpret_tensor(arg55_1, (192, 196), (1, 192), 0), out=buf94)
        del arg55_1
        buf95 = buf87; del buf87  # reuse
        buf96 = buf86; del buf86  # reuse
        buf97 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_349, layer_norm_58], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf84, buf94, arg56_1, buf95, buf96, buf97, 4704, 128, grid=grid(4704), stream=stream0)
        buf98 = buf89; del buf89  # reuse
        buf99 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_349, layer_norm_58], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf95, buf96, buf97, buf98, buf99, 1568, 3, grid=grid(1568), stream=stream0)
        buf101 = reinterpret_tensor(buf73, (8, 196, 384), (75264, 384, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_349, layer_norm_58], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf84, buf94, arg56_1, buf98, buf99, arg57_1, arg58_1, buf101, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg57_1
        del arg58_1
        buf102 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (1568, 384), (384, 1), 0), reinterpret_tensor(arg59_1, (384, 1536), (1, 384), 0), out=buf102)
        del arg59_1
        buf103 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [silu_57, x_351], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf102, arg60_1, buf103, 1204224, grid=grid(1204224), stream=stream0)
        del arg60_1
        buf104 = reinterpret_tensor(buf101, (1568, 384), (384, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (1568, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 384), (1, 768), 0), out=buf104)
        del arg61_1
        buf105 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_349, x_355], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf105, buf94, arg56_1, buf104, arg62_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg56_1
        del arg62_1
        buf106 = buf97; del buf97  # reuse
        buf107 = buf96; del buf96  # reuse
        buf108 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_59], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf105, buf106, buf107, buf108, 4704, 128, grid=grid(4704), stream=stream0)
        buf109 = buf99; del buf99  # reuse
        buf110 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_59], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf106, buf107, buf108, buf109, buf110, 1568, 3, grid=grid(1568), stream=stream0)
        buf112 = reinterpret_tensor(buf94, (8, 384, 196), (75264, 196, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_356], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf105, buf109, buf110, arg63_1, arg64_1, buf112, 602112, grid=grid(602112), stream=stream0)
        del arg63_1
        del arg64_1
        buf113 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_356], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (3072, 196), (196, 1), 0), reinterpret_tensor(arg65_1, (196, 384), (1, 196), 0), out=buf113)
        del arg65_1
        buf114 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [silu_58, x_357], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf113, arg66_1, buf114, 589824, grid=grid(589824), stream=stream0)
        del arg66_1
        buf115 = reinterpret_tensor(buf112, (3072, 196), (196, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (3072, 192), (192, 1), 0), reinterpret_tensor(arg67_1, (192, 196), (1, 192), 0), out=buf115)
        del arg67_1
        buf116 = buf108; del buf108  # reuse
        buf117 = buf107; del buf107  # reuse
        buf118 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_361, layer_norm_60], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf105, buf115, arg68_1, buf116, buf117, buf118, 4704, 128, grid=grid(4704), stream=stream0)
        buf119 = buf110; del buf110  # reuse
        buf120 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_361, layer_norm_60], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf116, buf117, buf118, buf119, buf120, 1568, 3, grid=grid(1568), stream=stream0)
        buf122 = reinterpret_tensor(buf104, (8, 196, 384), (75264, 384, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_361, layer_norm_60], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf105, buf115, arg68_1, buf119, buf120, arg69_1, arg70_1, buf122, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg69_1
        del arg70_1
        buf123 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (1568, 384), (384, 1), 0), reinterpret_tensor(arg71_1, (384, 1536), (1, 384), 0), out=buf123)
        del arg71_1
        buf124 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [silu_59, x_363], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf123, arg72_1, buf124, 1204224, grid=grid(1204224), stream=stream0)
        del arg72_1
        buf125 = reinterpret_tensor(buf122, (1568, 384), (384, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (1568, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 384), (1, 768), 0), out=buf125)
        del arg73_1
        buf126 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_361, x_367], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf126, buf115, arg68_1, buf125, arg74_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg68_1
        del arg74_1
        buf127 = buf118; del buf118  # reuse
        buf128 = buf117; del buf117  # reuse
        buf129 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_61], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf126, buf127, buf128, buf129, 4704, 128, grid=grid(4704), stream=stream0)
        buf130 = buf120; del buf120  # reuse
        buf131 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_61], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf127, buf128, buf129, buf130, buf131, 1568, 3, grid=grid(1568), stream=stream0)
        buf133 = reinterpret_tensor(buf125, (8, 384, 196), (75264, 196, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [x_368], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf126, buf130, buf131, arg75_1, arg76_1, buf133, 602112, grid=grid(602112), stream=stream0)
        del arg75_1
        del arg76_1
        buf134 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [x_368], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (3072, 196), (196, 1), 0), reinterpret_tensor(arg77_1, (196, 384), (1, 196), 0), out=buf134)
        del arg77_1
        buf135 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [silu_60, x_369], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf134, arg78_1, buf135, 589824, grid=grid(589824), stream=stream0)
        del arg78_1
        buf136 = reinterpret_tensor(buf133, (3072, 196), (196, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (3072, 192), (192, 1), 0), reinterpret_tensor(arg79_1, (192, 196), (1, 192), 0), out=buf136)
        del arg79_1
        buf137 = buf129; del buf129  # reuse
        buf138 = buf128; del buf128  # reuse
        buf139 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_373, layer_norm_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf126, buf136, arg80_1, buf137, buf138, buf139, 4704, 128, grid=grid(4704), stream=stream0)
        buf140 = buf131; del buf131  # reuse
        buf141 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_373, layer_norm_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf137, buf138, buf139, buf140, buf141, 1568, 3, grid=grid(1568), stream=stream0)
        buf143 = reinterpret_tensor(buf115, (8, 196, 384), (75264, 384, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_373, layer_norm_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf126, buf136, arg80_1, buf140, buf141, arg81_1, arg82_1, buf143, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg81_1
        del arg82_1
        buf144 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (1568, 384), (384, 1), 0), reinterpret_tensor(arg83_1, (384, 1536), (1, 384), 0), out=buf144)
        del arg83_1
        buf145 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [silu_61, x_375], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf144, arg84_1, buf145, 1204224, grid=grid(1204224), stream=stream0)
        del arg84_1
        buf146 = reinterpret_tensor(buf143, (1568, 384), (384, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (1568, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 384), (1, 768), 0), out=buf146)
        del arg85_1
        buf147 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_373, x_379], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf147, buf136, arg80_1, buf146, arg86_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg80_1
        del arg86_1
        buf148 = buf139; del buf139  # reuse
        buf149 = buf138; del buf138  # reuse
        buf150 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_63], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf147, buf148, buf149, buf150, 4704, 128, grid=grid(4704), stream=stream0)
        buf151 = buf141; del buf141  # reuse
        buf152 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_63], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf148, buf149, buf150, buf151, buf152, 1568, 3, grid=grid(1568), stream=stream0)
        buf154 = reinterpret_tensor(buf146, (8, 384, 196), (75264, 196, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_380], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf147, buf151, buf152, arg87_1, arg88_1, buf154, 602112, grid=grid(602112), stream=stream0)
        del arg87_1
        del arg88_1
        buf155 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [x_380], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (3072, 196), (196, 1), 0), reinterpret_tensor(arg89_1, (196, 384), (1, 196), 0), out=buf155)
        del arg89_1
        buf156 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [silu_62, x_381], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf155, arg90_1, buf156, 589824, grid=grid(589824), stream=stream0)
        del arg90_1
        buf157 = reinterpret_tensor(buf154, (3072, 196), (196, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (3072, 192), (192, 1), 0), reinterpret_tensor(arg91_1, (192, 196), (1, 192), 0), out=buf157)
        del arg91_1
        buf158 = buf150; del buf150  # reuse
        buf159 = buf149; del buf149  # reuse
        buf160 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_385, layer_norm_64], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf147, buf157, arg92_1, buf158, buf159, buf160, 4704, 128, grid=grid(4704), stream=stream0)
        buf161 = buf152; del buf152  # reuse
        buf162 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [x_385, layer_norm_64], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf158, buf159, buf160, buf161, buf162, 1568, 3, grid=grid(1568), stream=stream0)
        buf164 = reinterpret_tensor(buf136, (8, 196, 384), (75264, 384, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_385, layer_norm_64], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf147, buf157, arg92_1, buf161, buf162, arg93_1, arg94_1, buf164, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg93_1
        del arg94_1
        buf165 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (1568, 384), (384, 1), 0), reinterpret_tensor(arg95_1, (384, 1536), (1, 384), 0), out=buf165)
        del arg95_1
        buf166 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [silu_63, x_387], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf165, arg96_1, buf166, 1204224, grid=grid(1204224), stream=stream0)
        del arg96_1
        buf167 = reinterpret_tensor(buf164, (1568, 384), (384, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (1568, 768), (768, 1), 0), reinterpret_tensor(arg97_1, (768, 384), (1, 768), 0), out=buf167)
        del arg97_1
        buf168 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_385, x_391], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf168, buf157, arg92_1, buf167, arg98_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg92_1
        del arg98_1
        buf169 = buf160; del buf160  # reuse
        buf170 = buf159; del buf159  # reuse
        buf171 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_65], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf168, buf169, buf170, buf171, 4704, 128, grid=grid(4704), stream=stream0)
        buf172 = buf162; del buf162  # reuse
        buf173 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_65], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf169, buf170, buf171, buf172, buf173, 1568, 3, grid=grid(1568), stream=stream0)
        buf175 = reinterpret_tensor(buf167, (8, 384, 196), (75264, 196, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_392], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf168, buf172, buf173, arg99_1, arg100_1, buf175, 602112, grid=grid(602112), stream=stream0)
        del arg100_1
        del arg99_1
        buf176 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_392], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (3072, 196), (196, 1), 0), reinterpret_tensor(arg101_1, (196, 384), (1, 196), 0), out=buf176)
        del arg101_1
        buf177 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [silu_64, x_393], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf176, arg102_1, buf177, 589824, grid=grid(589824), stream=stream0)
        del arg102_1
        buf178 = reinterpret_tensor(buf175, (3072, 196), (196, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf177, (3072, 192), (192, 1), 0), reinterpret_tensor(arg103_1, (192, 196), (1, 192), 0), out=buf178)
        del arg103_1
        buf179 = buf171; del buf171  # reuse
        buf180 = buf170; del buf170  # reuse
        buf181 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_397, layer_norm_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf168, buf178, arg104_1, buf179, buf180, buf181, 4704, 128, grid=grid(4704), stream=stream0)
        buf182 = buf173; del buf173  # reuse
        buf183 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [x_397, layer_norm_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf179, buf180, buf181, buf182, buf183, 1568, 3, grid=grid(1568), stream=stream0)
        buf185 = reinterpret_tensor(buf157, (8, 196, 384), (75264, 384, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_397, layer_norm_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf168, buf178, arg104_1, buf182, buf183, arg105_1, arg106_1, buf185, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg105_1
        del arg106_1
        buf186 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf185, (1568, 384), (384, 1), 0), reinterpret_tensor(arg107_1, (384, 1536), (1, 384), 0), out=buf186)
        del arg107_1
        buf187 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [silu_65, x_399], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf186, arg108_1, buf187, 1204224, grid=grid(1204224), stream=stream0)
        del arg108_1
        buf188 = reinterpret_tensor(buf185, (1568, 384), (384, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (1568, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 384), (1, 768), 0), out=buf188)
        del arg109_1
        buf189 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_397, x_403], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf189, buf178, arg104_1, buf188, arg110_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg104_1
        del arg110_1
        buf190 = buf181; del buf181  # reuse
        buf191 = buf180; del buf180  # reuse
        buf192 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_67], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf189, buf190, buf191, buf192, 4704, 128, grid=grid(4704), stream=stream0)
        buf193 = buf183; del buf183  # reuse
        buf194 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_67], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf190, buf191, buf192, buf193, buf194, 1568, 3, grid=grid(1568), stream=stream0)
        buf196 = reinterpret_tensor(buf188, (8, 384, 196), (75264, 196, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_404], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf189, buf193, buf194, arg111_1, arg112_1, buf196, 602112, grid=grid(602112), stream=stream0)
        del arg111_1
        del arg112_1
        buf197 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [x_404], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (3072, 196), (196, 1), 0), reinterpret_tensor(arg113_1, (196, 384), (1, 196), 0), out=buf197)
        del arg113_1
        buf198 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [silu_66, x_405], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf197, arg114_1, buf198, 589824, grid=grid(589824), stream=stream0)
        del arg114_1
        buf199 = reinterpret_tensor(buf196, (3072, 196), (196, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (3072, 192), (192, 1), 0), reinterpret_tensor(arg115_1, (192, 196), (1, 192), 0), out=buf199)
        del arg115_1
        buf200 = buf192; del buf192  # reuse
        buf201 = buf191; del buf191  # reuse
        buf202 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_409, layer_norm_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf189, buf199, arg116_1, buf200, buf201, buf202, 4704, 128, grid=grid(4704), stream=stream0)
        buf203 = buf194; del buf194  # reuse
        buf204 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [x_409, layer_norm_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf200, buf201, buf202, buf203, buf204, 1568, 3, grid=grid(1568), stream=stream0)
        buf206 = reinterpret_tensor(buf178, (8, 196, 384), (75264, 384, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_409, layer_norm_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf189, buf199, arg116_1, buf203, buf204, arg117_1, arg118_1, buf206, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg117_1
        del arg118_1
        buf207 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (1568, 384), (384, 1), 0), reinterpret_tensor(arg119_1, (384, 1536), (1, 384), 0), out=buf207)
        del arg119_1
        buf208 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [silu_67, x_411], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf207, arg120_1, buf208, 1204224, grid=grid(1204224), stream=stream0)
        del arg120_1
        buf209 = reinterpret_tensor(buf206, (1568, 384), (384, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf208, (1568, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 384), (1, 768), 0), out=buf209)
        del arg121_1
        buf210 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_409, x_415], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf210, buf199, arg116_1, buf209, arg122_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg116_1
        del arg122_1
        buf211 = buf202; del buf202  # reuse
        buf212 = buf201; del buf201  # reuse
        buf213 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_69], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf210, buf211, buf212, buf213, 4704, 128, grid=grid(4704), stream=stream0)
        buf214 = buf204; del buf204  # reuse
        buf215 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_69], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf211, buf212, buf213, buf214, buf215, 1568, 3, grid=grid(1568), stream=stream0)
        buf217 = reinterpret_tensor(buf209, (8, 384, 196), (75264, 196, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [x_416], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf210, buf214, buf215, arg123_1, arg124_1, buf217, 602112, grid=grid(602112), stream=stream0)
        del arg123_1
        del arg124_1
        buf218 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [x_416], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (3072, 196), (196, 1), 0), reinterpret_tensor(arg125_1, (196, 384), (1, 196), 0), out=buf218)
        del arg125_1
        buf219 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [silu_68, x_417], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf218, arg126_1, buf219, 589824, grid=grid(589824), stream=stream0)
        del arg126_1
        buf220 = reinterpret_tensor(buf217, (3072, 196), (196, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (3072, 192), (192, 1), 0), reinterpret_tensor(arg127_1, (192, 196), (1, 192), 0), out=buf220)
        del arg127_1
        buf221 = buf213; del buf213  # reuse
        buf222 = buf212; del buf212  # reuse
        buf223 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [x_421, layer_norm_70], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf210, buf220, arg128_1, buf221, buf222, buf223, 4704, 128, grid=grid(4704), stream=stream0)
        buf224 = buf215; del buf215  # reuse
        buf225 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [x_421, layer_norm_70], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf221, buf222, buf223, buf224, buf225, 1568, 3, grid=grid(1568), stream=stream0)
        buf227 = reinterpret_tensor(buf199, (8, 196, 384), (75264, 384, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [x_421, layer_norm_70], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf210, buf220, arg128_1, buf224, buf225, arg129_1, arg130_1, buf227, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg129_1
        del arg130_1
        buf228 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf227, (1568, 384), (384, 1), 0), reinterpret_tensor(arg131_1, (384, 1536), (1, 384), 0), out=buf228)
        del arg131_1
        buf229 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [silu_69, x_423], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf228, arg132_1, buf229, 1204224, grid=grid(1204224), stream=stream0)
        del arg132_1
        buf230 = reinterpret_tensor(buf227, (1568, 384), (384, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (1568, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 384), (1, 768), 0), out=buf230)
        del arg133_1
        buf231 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [x_421, x_427], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf231, buf220, arg128_1, buf230, arg134_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg128_1
        del arg134_1
        buf232 = buf223; del buf223  # reuse
        buf233 = buf222; del buf222  # reuse
        buf234 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_71], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf231, buf232, buf233, buf234, 4704, 128, grid=grid(4704), stream=stream0)
        buf235 = buf225; del buf225  # reuse
        buf236 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_71], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf232, buf233, buf234, buf235, buf236, 1568, 3, grid=grid(1568), stream=stream0)
        buf238 = reinterpret_tensor(buf230, (8, 384, 196), (75264, 196, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [x_428], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf231, buf235, buf236, arg135_1, arg136_1, buf238, 602112, grid=grid(602112), stream=stream0)
        del arg135_1
        del arg136_1
        buf239 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [x_428], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (3072, 196), (196, 1), 0), reinterpret_tensor(arg137_1, (196, 384), (1, 196), 0), out=buf239)
        del arg137_1
        buf240 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [silu_70, x_429], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf239, arg138_1, buf240, 589824, grid=grid(589824), stream=stream0)
        del arg138_1
        buf241 = reinterpret_tensor(buf238, (3072, 196), (196, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (3072, 192), (192, 1), 0), reinterpret_tensor(arg139_1, (192, 196), (1, 192), 0), out=buf241)
        del arg139_1
        buf242 = buf234; del buf234  # reuse
        buf243 = buf233; del buf233  # reuse
        buf244 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_433, layer_norm_72], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf231, buf241, arg140_1, buf242, buf243, buf244, 4704, 128, grid=grid(4704), stream=stream0)
        buf245 = buf236; del buf236  # reuse
        buf246 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [x_433, layer_norm_72], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf242, buf243, buf244, buf245, buf246, 1568, 3, grid=grid(1568), stream=stream0)
        buf248 = reinterpret_tensor(buf220, (8, 196, 384), (75264, 384, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [x_433, layer_norm_72], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf231, buf241, arg140_1, buf245, buf246, arg141_1, arg142_1, buf248, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg141_1
        del arg142_1
        buf249 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (1568, 384), (384, 1), 0), reinterpret_tensor(arg143_1, (384, 1536), (1, 384), 0), out=buf249)
        del arg143_1
        buf250 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [silu_71, x_435], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf249, arg144_1, buf250, 1204224, grid=grid(1204224), stream=stream0)
        del arg144_1
        buf251 = reinterpret_tensor(buf248, (1568, 384), (384, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (1568, 768), (768, 1), 0), reinterpret_tensor(arg145_1, (768, 384), (1, 768), 0), out=buf251)
        del arg145_1
        buf252 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [x_433, x_439], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf252, buf241, arg140_1, buf251, arg146_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg140_1
        del arg146_1
        buf253 = buf244; del buf244  # reuse
        buf254 = buf243; del buf243  # reuse
        buf255 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_73], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf252, buf253, buf254, buf255, 4704, 128, grid=grid(4704), stream=stream0)
        buf256 = buf246; del buf246  # reuse
        buf257 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_73], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf253, buf254, buf255, buf256, buf257, 1568, 3, grid=grid(1568), stream=stream0)
        buf259 = reinterpret_tensor(buf251, (8, 384, 196), (75264, 196, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [x_440], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf252, buf256, buf257, arg147_1, arg148_1, buf259, 602112, grid=grid(602112), stream=stream0)
        del arg147_1
        del arg148_1
        buf260 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [x_440], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (3072, 196), (196, 1), 0), reinterpret_tensor(arg149_1, (196, 384), (1, 196), 0), out=buf260)
        del arg149_1
        buf261 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [silu_72, x_441], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf260, arg150_1, buf261, 589824, grid=grid(589824), stream=stream0)
        del arg150_1
        buf262 = reinterpret_tensor(buf259, (3072, 196), (196, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf261, (3072, 192), (192, 1), 0), reinterpret_tensor(arg151_1, (192, 196), (1, 192), 0), out=buf262)
        del arg151_1
        buf263 = buf255; del buf255  # reuse
        buf264 = buf254; del buf254  # reuse
        buf265 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [x_445, layer_norm_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf252, buf262, arg152_1, buf263, buf264, buf265, 4704, 128, grid=grid(4704), stream=stream0)
        buf266 = buf257; del buf257  # reuse
        buf267 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [x_445, layer_norm_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf263, buf264, buf265, buf266, buf267, 1568, 3, grid=grid(1568), stream=stream0)
        buf269 = reinterpret_tensor(buf241, (8, 196, 384), (75264, 384, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [x_445, layer_norm_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf252, buf262, arg152_1, buf266, buf267, arg153_1, arg154_1, buf269, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg153_1
        del arg154_1
        buf270 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (1568, 384), (384, 1), 0), reinterpret_tensor(arg155_1, (384, 1536), (1, 384), 0), out=buf270)
        del arg155_1
        buf271 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [silu_73, x_447], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf270, arg156_1, buf271, 1204224, grid=grid(1204224), stream=stream0)
        del arg156_1
        buf272 = reinterpret_tensor(buf269, (1568, 384), (384, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (1568, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 384), (1, 768), 0), out=buf272)
        del arg157_1
        buf273 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [x_445, x_451], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf273, buf262, arg152_1, buf272, arg158_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg152_1
        del arg158_1
        buf274 = buf265; del buf265  # reuse
        buf275 = buf264; del buf264  # reuse
        buf276 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_75], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf273, buf274, buf275, buf276, 4704, 128, grid=grid(4704), stream=stream0)
        buf277 = buf267; del buf267  # reuse
        buf278 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_75], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf274, buf275, buf276, buf277, buf278, 1568, 3, grid=grid(1568), stream=stream0)
        buf280 = reinterpret_tensor(buf272, (8, 384, 196), (75264, 196, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [x_452], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf273, buf277, buf278, arg159_1, arg160_1, buf280, 602112, grid=grid(602112), stream=stream0)
        del arg159_1
        del arg160_1
        buf281 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [x_452], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (3072, 196), (196, 1), 0), reinterpret_tensor(arg161_1, (196, 384), (1, 196), 0), out=buf281)
        del arg161_1
        buf282 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [silu_74, x_453], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf281, arg162_1, buf282, 589824, grid=grid(589824), stream=stream0)
        del arg162_1
        buf283 = reinterpret_tensor(buf280, (3072, 196), (196, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf282, (3072, 192), (192, 1), 0), reinterpret_tensor(arg163_1, (192, 196), (1, 192), 0), out=buf283)
        del arg163_1
        buf284 = buf276; del buf276  # reuse
        buf285 = buf275; del buf275  # reuse
        buf286 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [x_457, layer_norm_76], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf273, buf283, arg164_1, buf284, buf285, buf286, 4704, 128, grid=grid(4704), stream=stream0)
        buf287 = buf278; del buf278  # reuse
        buf288 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [x_457, layer_norm_76], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf284, buf285, buf286, buf287, buf288, 1568, 3, grid=grid(1568), stream=stream0)
        buf290 = reinterpret_tensor(buf262, (8, 196, 384), (75264, 384, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [x_457, layer_norm_76], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf273, buf283, arg164_1, buf287, buf288, arg165_1, arg166_1, buf290, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg165_1
        del arg166_1
        buf291 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf290, (1568, 384), (384, 1), 0), reinterpret_tensor(arg167_1, (384, 1536), (1, 384), 0), out=buf291)
        del arg167_1
        buf292 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [silu_75, x_459], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf291, arg168_1, buf292, 1204224, grid=grid(1204224), stream=stream0)
        del arg168_1
        buf293 = reinterpret_tensor(buf290, (1568, 384), (384, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (1568, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 384), (1, 768), 0), out=buf293)
        del arg169_1
        buf294 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [x_457, x_463], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf294, buf283, arg164_1, buf293, arg170_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg164_1
        del arg170_1
        buf295 = buf286; del buf286  # reuse
        buf296 = buf285; del buf285  # reuse
        buf297 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_77], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf294, buf295, buf296, buf297, 4704, 128, grid=grid(4704), stream=stream0)
        buf298 = buf288; del buf288  # reuse
        buf299 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_77], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf295, buf296, buf297, buf298, buf299, 1568, 3, grid=grid(1568), stream=stream0)
        buf301 = reinterpret_tensor(buf293, (8, 384, 196), (75264, 196, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [x_464], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf294, buf298, buf299, arg171_1, arg172_1, buf301, 602112, grid=grid(602112), stream=stream0)
        del arg171_1
        del arg172_1
        buf302 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [x_464], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf301, (3072, 196), (196, 1), 0), reinterpret_tensor(arg173_1, (196, 384), (1, 196), 0), out=buf302)
        del arg173_1
        buf303 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [silu_76, x_465], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf302, arg174_1, buf303, 589824, grid=grid(589824), stream=stream0)
        del arg174_1
        buf304 = reinterpret_tensor(buf301, (3072, 196), (196, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (3072, 192), (192, 1), 0), reinterpret_tensor(arg175_1, (192, 196), (1, 192), 0), out=buf304)
        del arg175_1
        buf305 = buf297; del buf297  # reuse
        buf306 = buf296; del buf296  # reuse
        buf307 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [x_469, layer_norm_78], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf294, buf304, arg176_1, buf305, buf306, buf307, 4704, 128, grid=grid(4704), stream=stream0)
        buf308 = buf299; del buf299  # reuse
        buf309 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [x_469, layer_norm_78], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf305, buf306, buf307, buf308, buf309, 1568, 3, grid=grid(1568), stream=stream0)
        buf311 = reinterpret_tensor(buf283, (8, 196, 384), (75264, 384, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [x_469, layer_norm_78], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf294, buf304, arg176_1, buf308, buf309, arg177_1, arg178_1, buf311, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg177_1
        del arg178_1
        buf312 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf311, (1568, 384), (384, 1), 0), reinterpret_tensor(arg179_1, (384, 1536), (1, 384), 0), out=buf312)
        del arg179_1
        buf313 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [silu_77, x_471], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf312, arg180_1, buf313, 1204224, grid=grid(1204224), stream=stream0)
        del arg180_1
        buf314 = reinterpret_tensor(buf311, (1568, 384), (384, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (1568, 768), (768, 1), 0), reinterpret_tensor(arg181_1, (768, 384), (1, 768), 0), out=buf314)
        del arg181_1
        buf315 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [x_469, x_475], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf315, buf304, arg176_1, buf314, arg182_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg176_1
        del arg182_1
        buf316 = buf307; del buf307  # reuse
        buf317 = buf306; del buf306  # reuse
        buf318 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_79], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf315, buf316, buf317, buf318, 4704, 128, grid=grid(4704), stream=stream0)
        buf319 = buf309; del buf309  # reuse
        buf320 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_79], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf316, buf317, buf318, buf319, buf320, 1568, 3, grid=grid(1568), stream=stream0)
        buf322 = reinterpret_tensor(buf314, (8, 384, 196), (75264, 196, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [x_476], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf315, buf319, buf320, arg183_1, arg184_1, buf322, 602112, grid=grid(602112), stream=stream0)
        del arg183_1
        del arg184_1
        buf323 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [x_476], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (3072, 196), (196, 1), 0), reinterpret_tensor(arg185_1, (196, 384), (1, 196), 0), out=buf323)
        del arg185_1
        buf324 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [silu_78, x_477], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf323, arg186_1, buf324, 589824, grid=grid(589824), stream=stream0)
        del arg186_1
        buf325 = reinterpret_tensor(buf322, (3072, 196), (196, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf324, (3072, 192), (192, 1), 0), reinterpret_tensor(arg187_1, (192, 196), (1, 192), 0), out=buf325)
        del arg187_1
        buf326 = buf318; del buf318  # reuse
        buf327 = buf317; del buf317  # reuse
        buf328 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [x_481, layer_norm_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf315, buf325, arg188_1, buf326, buf327, buf328, 4704, 128, grid=grid(4704), stream=stream0)
        buf329 = buf320; del buf320  # reuse
        buf330 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [x_481, layer_norm_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf326, buf327, buf328, buf329, buf330, 1568, 3, grid=grid(1568), stream=stream0)
        buf332 = reinterpret_tensor(buf304, (8, 196, 384), (75264, 384, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [x_481, layer_norm_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf315, buf325, arg188_1, buf329, buf330, arg189_1, arg190_1, buf332, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg189_1
        del arg190_1
        buf333 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (1568, 384), (384, 1), 0), reinterpret_tensor(arg191_1, (384, 1536), (1, 384), 0), out=buf333)
        del arg191_1
        buf334 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [silu_79, x_483], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf333, arg192_1, buf334, 1204224, grid=grid(1204224), stream=stream0)
        del arg192_1
        buf335 = reinterpret_tensor(buf332, (1568, 384), (384, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (1568, 768), (768, 1), 0), reinterpret_tensor(arg193_1, (768, 384), (1, 768), 0), out=buf335)
        del arg193_1
        buf336 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [x_481, x_487], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf336, buf325, arg188_1, buf335, arg194_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg188_1
        del arg194_1
        buf337 = buf328; del buf328  # reuse
        buf338 = buf327; del buf327  # reuse
        buf339 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_81], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf336, buf337, buf338, buf339, 4704, 128, grid=grid(4704), stream=stream0)
        buf340 = buf330; del buf330  # reuse
        buf341 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_81], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf337, buf338, buf339, buf340, buf341, 1568, 3, grid=grid(1568), stream=stream0)
        buf343 = reinterpret_tensor(buf335, (8, 384, 196), (75264, 196, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [x_488], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf336, buf340, buf341, arg195_1, arg196_1, buf343, 602112, grid=grid(602112), stream=stream0)
        del arg195_1
        del arg196_1
        buf344 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [x_488], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf343, (3072, 196), (196, 1), 0), reinterpret_tensor(arg197_1, (196, 384), (1, 196), 0), out=buf344)
        del arg197_1
        buf345 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [silu_80, x_489], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf344, arg198_1, buf345, 589824, grid=grid(589824), stream=stream0)
        del arg198_1
        buf346 = reinterpret_tensor(buf343, (3072, 196), (196, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf345, (3072, 192), (192, 1), 0), reinterpret_tensor(arg199_1, (192, 196), (1, 192), 0), out=buf346)
        del arg199_1
        buf347 = buf339; del buf339  # reuse
        buf348 = buf338; del buf338  # reuse
        buf349 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [x_493, layer_norm_82], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf336, buf346, arg200_1, buf347, buf348, buf349, 4704, 128, grid=grid(4704), stream=stream0)
        buf350 = buf341; del buf341  # reuse
        buf351 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [x_493, layer_norm_82], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf347, buf348, buf349, buf350, buf351, 1568, 3, grid=grid(1568), stream=stream0)
        buf353 = reinterpret_tensor(buf325, (8, 196, 384), (75264, 384, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [x_493, layer_norm_82], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf336, buf346, arg200_1, buf350, buf351, arg201_1, arg202_1, buf353, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg201_1
        del arg202_1
        buf354 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (1568, 384), (384, 1), 0), reinterpret_tensor(arg203_1, (384, 1536), (1, 384), 0), out=buf354)
        del arg203_1
        buf355 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [silu_81, x_495], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf354, arg204_1, buf355, 1204224, grid=grid(1204224), stream=stream0)
        del arg204_1
        buf356 = reinterpret_tensor(buf353, (1568, 384), (384, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf355, (1568, 768), (768, 1), 0), reinterpret_tensor(arg205_1, (768, 384), (1, 768), 0), out=buf356)
        del arg205_1
        buf357 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [x_493, x_499], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf357, buf346, arg200_1, buf356, arg206_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg200_1
        del arg206_1
        buf358 = buf349; del buf349  # reuse
        buf359 = buf348; del buf348  # reuse
        buf360 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_83], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf357, buf358, buf359, buf360, 4704, 128, grid=grid(4704), stream=stream0)
        buf361 = buf351; del buf351  # reuse
        buf362 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_83], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf358, buf359, buf360, buf361, buf362, 1568, 3, grid=grid(1568), stream=stream0)
        buf364 = reinterpret_tensor(buf356, (8, 384, 196), (75264, 196, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [x_500], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf357, buf361, buf362, arg207_1, arg208_1, buf364, 602112, grid=grid(602112), stream=stream0)
        del arg207_1
        del arg208_1
        buf365 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [x_500], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (3072, 196), (196, 1), 0), reinterpret_tensor(arg209_1, (196, 384), (1, 196), 0), out=buf365)
        del arg209_1
        buf366 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [silu_82, x_501], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf365, arg210_1, buf366, 589824, grid=grid(589824), stream=stream0)
        del arg210_1
        buf367 = reinterpret_tensor(buf364, (3072, 196), (196, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf366, (3072, 192), (192, 1), 0), reinterpret_tensor(arg211_1, (192, 196), (1, 192), 0), out=buf367)
        del arg211_1
        buf368 = buf360; del buf360  # reuse
        buf369 = buf359; del buf359  # reuse
        buf370 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [x_505, layer_norm_84], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf357, buf367, arg212_1, buf368, buf369, buf370, 4704, 128, grid=grid(4704), stream=stream0)
        buf371 = buf362; del buf362  # reuse
        buf372 = buf361; del buf361  # reuse
        # Topologically Sorted Source Nodes: [x_505, layer_norm_84], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf368, buf369, buf370, buf371, buf372, 1568, 3, grid=grid(1568), stream=stream0)
        buf374 = reinterpret_tensor(buf346, (8, 196, 384), (75264, 384, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [x_505, layer_norm_84], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf357, buf367, arg212_1, buf371, buf372, arg213_1, arg214_1, buf374, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg213_1
        del arg214_1
        buf375 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf374, (1568, 384), (384, 1), 0), reinterpret_tensor(arg215_1, (384, 1536), (1, 384), 0), out=buf375)
        del arg215_1
        buf376 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [silu_83, x_507], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf375, arg216_1, buf376, 1204224, grid=grid(1204224), stream=stream0)
        del arg216_1
        buf377 = reinterpret_tensor(buf374, (1568, 384), (384, 1), 0); del buf374  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf376, (1568, 768), (768, 1), 0), reinterpret_tensor(arg217_1, (768, 384), (1, 768), 0), out=buf377)
        del arg217_1
        buf378 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [x_505, x_511], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf378, buf367, arg212_1, buf377, arg218_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg212_1
        del arg218_1
        buf379 = buf370; del buf370  # reuse
        buf380 = buf369; del buf369  # reuse
        buf381 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_85], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf378, buf379, buf380, buf381, 4704, 128, grid=grid(4704), stream=stream0)
        buf382 = buf372; del buf372  # reuse
        buf383 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_85], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf379, buf380, buf381, buf382, buf383, 1568, 3, grid=grid(1568), stream=stream0)
        buf385 = reinterpret_tensor(buf377, (8, 384, 196), (75264, 196, 1), 0); del buf377  # reuse
        # Topologically Sorted Source Nodes: [x_512], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf378, buf382, buf383, arg219_1, arg220_1, buf385, 602112, grid=grid(602112), stream=stream0)
        del arg219_1
        del arg220_1
        buf386 = buf365; del buf365  # reuse
        # Topologically Sorted Source Nodes: [x_512], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf385, (3072, 196), (196, 1), 0), reinterpret_tensor(arg221_1, (196, 384), (1, 196), 0), out=buf386)
        del arg221_1
        buf387 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [silu_84, x_513], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf386, arg222_1, buf387, 589824, grid=grid(589824), stream=stream0)
        del arg222_1
        buf388 = reinterpret_tensor(buf385, (3072, 196), (196, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf387, (3072, 192), (192, 1), 0), reinterpret_tensor(arg223_1, (192, 196), (1, 192), 0), out=buf388)
        del arg223_1
        buf389 = buf381; del buf381  # reuse
        buf390 = buf380; del buf380  # reuse
        buf391 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [x_517, layer_norm_86], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf378, buf388, arg224_1, buf389, buf390, buf391, 4704, 128, grid=grid(4704), stream=stream0)
        buf392 = buf383; del buf383  # reuse
        buf393 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [x_517, layer_norm_86], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf389, buf390, buf391, buf392, buf393, 1568, 3, grid=grid(1568), stream=stream0)
        buf395 = reinterpret_tensor(buf367, (8, 196, 384), (75264, 384, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [x_517, layer_norm_86], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf378, buf388, arg224_1, buf392, buf393, arg225_1, arg226_1, buf395, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg225_1
        del arg226_1
        buf396 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf395, (1568, 384), (384, 1), 0), reinterpret_tensor(arg227_1, (384, 1536), (1, 384), 0), out=buf396)
        del arg227_1
        buf397 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [silu_85, x_519], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf396, arg228_1, buf397, 1204224, grid=grid(1204224), stream=stream0)
        del arg228_1
        buf398 = reinterpret_tensor(buf395, (1568, 384), (384, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf397, (1568, 768), (768, 1), 0), reinterpret_tensor(arg229_1, (768, 384), (1, 768), 0), out=buf398)
        del arg229_1
        buf399 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [x_517, x_523], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf399, buf388, arg224_1, buf398, arg230_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg224_1
        del arg230_1
        buf400 = buf391; del buf391  # reuse
        buf401 = buf390; del buf390  # reuse
        buf402 = buf389; del buf389  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_87], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf399, buf400, buf401, buf402, 4704, 128, grid=grid(4704), stream=stream0)
        buf403 = buf393; del buf393  # reuse
        buf404 = buf392; del buf392  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_87], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf400, buf401, buf402, buf403, buf404, 1568, 3, grid=grid(1568), stream=stream0)
        buf406 = reinterpret_tensor(buf398, (8, 384, 196), (75264, 196, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [x_524], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf399, buf403, buf404, arg231_1, arg232_1, buf406, 602112, grid=grid(602112), stream=stream0)
        del arg231_1
        del arg232_1
        buf407 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [x_524], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (3072, 196), (196, 1), 0), reinterpret_tensor(arg233_1, (196, 384), (1, 196), 0), out=buf407)
        del arg233_1
        buf408 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [silu_86, x_525], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf407, arg234_1, buf408, 589824, grid=grid(589824), stream=stream0)
        del arg234_1
        buf409 = reinterpret_tensor(buf406, (3072, 196), (196, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf408, (3072, 192), (192, 1), 0), reinterpret_tensor(arg235_1, (192, 196), (1, 192), 0), out=buf409)
        del arg235_1
        buf410 = buf402; del buf402  # reuse
        buf411 = buf401; del buf401  # reuse
        buf412 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [x_529, layer_norm_88], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf399, buf409, arg236_1, buf410, buf411, buf412, 4704, 128, grid=grid(4704), stream=stream0)
        buf413 = buf404; del buf404  # reuse
        buf414 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [x_529, layer_norm_88], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf410, buf411, buf412, buf413, buf414, 1568, 3, grid=grid(1568), stream=stream0)
        buf416 = reinterpret_tensor(buf388, (8, 196, 384), (75264, 384, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [x_529, layer_norm_88], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf399, buf409, arg236_1, buf413, buf414, arg237_1, arg238_1, buf416, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg237_1
        del arg238_1
        buf417 = buf396; del buf396  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf416, (1568, 384), (384, 1), 0), reinterpret_tensor(arg239_1, (384, 1536), (1, 384), 0), out=buf417)
        del arg239_1
        buf418 = buf397; del buf397  # reuse
        # Topologically Sorted Source Nodes: [silu_87, x_531], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf417, arg240_1, buf418, 1204224, grid=grid(1204224), stream=stream0)
        del arg240_1
        buf419 = reinterpret_tensor(buf416, (1568, 384), (384, 1), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf418, (1568, 768), (768, 1), 0), reinterpret_tensor(arg241_1, (768, 384), (1, 768), 0), out=buf419)
        del arg241_1
        buf420 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [x_529, x_535], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf420, buf409, arg236_1, buf419, arg242_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg236_1
        del arg242_1
        buf421 = buf412; del buf412  # reuse
        buf422 = buf411; del buf411  # reuse
        buf423 = buf410; del buf410  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_89], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf420, buf421, buf422, buf423, 4704, 128, grid=grid(4704), stream=stream0)
        buf424 = buf414; del buf414  # reuse
        buf425 = buf413; del buf413  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_89], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf421, buf422, buf423, buf424, buf425, 1568, 3, grid=grid(1568), stream=stream0)
        buf427 = reinterpret_tensor(buf419, (8, 384, 196), (75264, 196, 1), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [x_536], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf420, buf424, buf425, arg243_1, arg244_1, buf427, 602112, grid=grid(602112), stream=stream0)
        del arg243_1
        del arg244_1
        buf428 = buf407; del buf407  # reuse
        # Topologically Sorted Source Nodes: [x_536], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf427, (3072, 196), (196, 1), 0), reinterpret_tensor(arg245_1, (196, 384), (1, 196), 0), out=buf428)
        del arg245_1
        buf429 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [silu_88, x_537], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf428, arg246_1, buf429, 589824, grid=grid(589824), stream=stream0)
        del arg246_1
        buf430 = reinterpret_tensor(buf427, (3072, 196), (196, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf429, (3072, 192), (192, 1), 0), reinterpret_tensor(arg247_1, (192, 196), (1, 192), 0), out=buf430)
        del arg247_1
        buf431 = buf423; del buf423  # reuse
        buf432 = buf422; del buf422  # reuse
        buf433 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [x_541, layer_norm_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf420, buf430, arg248_1, buf431, buf432, buf433, 4704, 128, grid=grid(4704), stream=stream0)
        buf434 = buf425; del buf425  # reuse
        buf435 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [x_541, layer_norm_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf431, buf432, buf433, buf434, buf435, 1568, 3, grid=grid(1568), stream=stream0)
        buf437 = reinterpret_tensor(buf409, (8, 196, 384), (75264, 384, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [x_541, layer_norm_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf420, buf430, arg248_1, buf434, buf435, arg249_1, arg250_1, buf437, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg249_1
        del arg250_1
        buf438 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf437, (1568, 384), (384, 1), 0), reinterpret_tensor(arg251_1, (384, 1536), (1, 384), 0), out=buf438)
        del arg251_1
        buf439 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [silu_89, x_543], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf438, arg252_1, buf439, 1204224, grid=grid(1204224), stream=stream0)
        del arg252_1
        buf440 = reinterpret_tensor(buf437, (1568, 384), (384, 1), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf439, (1568, 768), (768, 1), 0), reinterpret_tensor(arg253_1, (768, 384), (1, 768), 0), out=buf440)
        del arg253_1
        buf441 = buf420; del buf420  # reuse
        # Topologically Sorted Source Nodes: [x_541, x_547], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf441, buf430, arg248_1, buf440, arg254_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg248_1
        del arg254_1
        buf442 = buf433; del buf433  # reuse
        buf443 = buf432; del buf432  # reuse
        buf444 = buf431; del buf431  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_91], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf441, buf442, buf443, buf444, 4704, 128, grid=grid(4704), stream=stream0)
        buf445 = buf435; del buf435  # reuse
        buf446 = buf434; del buf434  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_91], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf442, buf443, buf444, buf445, buf446, 1568, 3, grid=grid(1568), stream=stream0)
        buf448 = reinterpret_tensor(buf440, (8, 384, 196), (75264, 196, 1), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [x_548], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf441, buf445, buf446, arg255_1, arg256_1, buf448, 602112, grid=grid(602112), stream=stream0)
        del arg255_1
        del arg256_1
        buf449 = buf428; del buf428  # reuse
        # Topologically Sorted Source Nodes: [x_548], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf448, (3072, 196), (196, 1), 0), reinterpret_tensor(arg257_1, (196, 384), (1, 196), 0), out=buf449)
        del arg257_1
        buf450 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [silu_90, x_549], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf449, arg258_1, buf450, 589824, grid=grid(589824), stream=stream0)
        del arg258_1
        buf451 = reinterpret_tensor(buf448, (3072, 196), (196, 1), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf450, (3072, 192), (192, 1), 0), reinterpret_tensor(arg259_1, (192, 196), (1, 192), 0), out=buf451)
        del arg259_1
        buf452 = buf444; del buf444  # reuse
        buf453 = buf443; del buf443  # reuse
        buf454 = buf442; del buf442  # reuse
        # Topologically Sorted Source Nodes: [x_553, layer_norm_92], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf441, buf451, arg260_1, buf452, buf453, buf454, 4704, 128, grid=grid(4704), stream=stream0)
        buf455 = buf446; del buf446  # reuse
        buf456 = buf445; del buf445  # reuse
        # Topologically Sorted Source Nodes: [x_553, layer_norm_92], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf452, buf453, buf454, buf455, buf456, 1568, 3, grid=grid(1568), stream=stream0)
        buf458 = reinterpret_tensor(buf430, (8, 196, 384), (75264, 384, 1), 0); del buf430  # reuse
        # Topologically Sorted Source Nodes: [x_553, layer_norm_92], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf441, buf451, arg260_1, buf455, buf456, arg261_1, arg262_1, buf458, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg261_1
        del arg262_1
        buf459 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf458, (1568, 384), (384, 1), 0), reinterpret_tensor(arg263_1, (384, 1536), (1, 384), 0), out=buf459)
        del arg263_1
        buf460 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [silu_91, x_555], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf459, arg264_1, buf460, 1204224, grid=grid(1204224), stream=stream0)
        del arg264_1
        buf461 = reinterpret_tensor(buf458, (1568, 384), (384, 1), 0); del buf458  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf460, (1568, 768), (768, 1), 0), reinterpret_tensor(arg265_1, (768, 384), (1, 768), 0), out=buf461)
        del arg265_1
        buf462 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [x_553, x_559], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf462, buf451, arg260_1, buf461, arg266_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg260_1
        del arg266_1
        buf463 = buf454; del buf454  # reuse
        buf464 = buf453; del buf453  # reuse
        buf465 = buf452; del buf452  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_93], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf462, buf463, buf464, buf465, 4704, 128, grid=grid(4704), stream=stream0)
        buf466 = buf456; del buf456  # reuse
        buf467 = buf455; del buf455  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_93], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf463, buf464, buf465, buf466, buf467, 1568, 3, grid=grid(1568), stream=stream0)
        buf469 = reinterpret_tensor(buf461, (8, 384, 196), (75264, 196, 1), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [x_560], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf462, buf466, buf467, arg267_1, arg268_1, buf469, 602112, grid=grid(602112), stream=stream0)
        del arg267_1
        del arg268_1
        buf470 = buf449; del buf449  # reuse
        # Topologically Sorted Source Nodes: [x_560], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf469, (3072, 196), (196, 1), 0), reinterpret_tensor(arg269_1, (196, 384), (1, 196), 0), out=buf470)
        del arg269_1
        buf471 = buf450; del buf450  # reuse
        # Topologically Sorted Source Nodes: [silu_92, x_561], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf470, arg270_1, buf471, 589824, grid=grid(589824), stream=stream0)
        del arg270_1
        buf472 = reinterpret_tensor(buf469, (3072, 196), (196, 1), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf471, (3072, 192), (192, 1), 0), reinterpret_tensor(arg271_1, (192, 196), (1, 192), 0), out=buf472)
        del arg271_1
        buf473 = buf465; del buf465  # reuse
        buf474 = buf464; del buf464  # reuse
        buf475 = buf463; del buf463  # reuse
        # Topologically Sorted Source Nodes: [x_565, layer_norm_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf462, buf472, arg272_1, buf473, buf474, buf475, 4704, 128, grid=grid(4704), stream=stream0)
        buf476 = buf467; del buf467  # reuse
        buf477 = buf466; del buf466  # reuse
        # Topologically Sorted Source Nodes: [x_565, layer_norm_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf473, buf474, buf475, buf476, buf477, 1568, 3, grid=grid(1568), stream=stream0)
        buf479 = reinterpret_tensor(buf451, (8, 196, 384), (75264, 384, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [x_565, layer_norm_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf462, buf472, arg272_1, buf476, buf477, arg273_1, arg274_1, buf479, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg273_1
        del arg274_1
        buf480 = buf459; del buf459  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf479, (1568, 384), (384, 1), 0), reinterpret_tensor(arg275_1, (384, 1536), (1, 384), 0), out=buf480)
        del arg275_1
        buf481 = buf460; del buf460  # reuse
        # Topologically Sorted Source Nodes: [silu_93, x_567], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf480, arg276_1, buf481, 1204224, grid=grid(1204224), stream=stream0)
        del arg276_1
        buf482 = reinterpret_tensor(buf479, (1568, 384), (384, 1), 0); del buf479  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf481, (1568, 768), (768, 1), 0), reinterpret_tensor(arg277_1, (768, 384), (1, 768), 0), out=buf482)
        del arg277_1
        buf483 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [x_565, x_571], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf483, buf472, arg272_1, buf482, arg278_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg272_1
        del arg278_1
        buf484 = buf475; del buf475  # reuse
        buf485 = buf474; del buf474  # reuse
        buf486 = buf473; del buf473  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_95], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf483, buf484, buf485, buf486, 4704, 128, grid=grid(4704), stream=stream0)
        buf487 = buf477; del buf477  # reuse
        buf488 = buf476; del buf476  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_95], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf484, buf485, buf486, buf487, buf488, 1568, 3, grid=grid(1568), stream=stream0)
        buf490 = reinterpret_tensor(buf482, (8, 384, 196), (75264, 196, 1), 0); del buf482  # reuse
        # Topologically Sorted Source Nodes: [x_572], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf483, buf487, buf488, arg279_1, arg280_1, buf490, 602112, grid=grid(602112), stream=stream0)
        del arg279_1
        del arg280_1
        buf491 = buf470; del buf470  # reuse
        # Topologically Sorted Source Nodes: [x_572], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (3072, 196), (196, 1), 0), reinterpret_tensor(arg281_1, (196, 384), (1, 196), 0), out=buf491)
        del arg281_1
        buf492 = buf471; del buf471  # reuse
        # Topologically Sorted Source Nodes: [silu_94, x_573], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_3.run(buf491, arg282_1, buf492, 589824, grid=grid(589824), stream=stream0)
        del arg282_1
        del buf491
        buf493 = reinterpret_tensor(buf490, (3072, 196), (196, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf492, (3072, 192), (192, 1), 0), reinterpret_tensor(arg283_1, (192, 196), (1, 192), 0), out=buf493)
        del arg283_1
        del buf492
        buf494 = buf486; del buf486  # reuse
        buf495 = buf485; del buf485  # reuse
        buf496 = buf484; del buf484  # reuse
        # Topologically Sorted Source Nodes: [x_577, layer_norm_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf483, buf493, arg284_1, buf494, buf495, buf496, 4704, 128, grid=grid(4704), stream=stream0)
        buf497 = buf488; del buf488  # reuse
        buf498 = buf487; del buf487  # reuse
        # Topologically Sorted Source Nodes: [x_577, layer_norm_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf494, buf495, buf496, buf497, buf498, 1568, 3, grid=grid(1568), stream=stream0)
        buf500 = reinterpret_tensor(buf472, (8, 196, 384), (75264, 384, 1), 0); del buf472  # reuse
        # Topologically Sorted Source Nodes: [x_577, layer_norm_96], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf483, buf493, arg284_1, buf497, buf498, arg285_1, arg286_1, buf500, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg285_1
        del arg286_1
        buf501 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf500, (1568, 384), (384, 1), 0), reinterpret_tensor(arg287_1, (384, 1536), (1, 384), 0), out=buf501)
        del arg287_1
        buf502 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [silu_95, x_579], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf501, arg288_1, buf502, 1204224, grid=grid(1204224), stream=stream0)
        del arg288_1
        del buf501
        buf503 = reinterpret_tensor(buf500, (1568, 384), (384, 1), 0); del buf500  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf502, (1568, 768), (768, 1), 0), reinterpret_tensor(arg289_1, (768, 384), (1, 768), 0), out=buf503)
        del arg289_1
        del buf502
        buf504 = buf483; del buf483  # reuse
        # Topologically Sorted Source Nodes: [x_577, x_583, x_584], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_12.run(buf504, buf493, arg284_1, buf503, arg290_1, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del arg284_1
        del arg290_1
        del buf493
        del buf503
        buf505 = buf496; del buf496  # reuse
        buf506 = buf495; del buf495  # reuse
        buf507 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [x_584], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf504, buf505, buf506, buf507, 4704, 128, grid=grid(4704), stream=stream0)
        buf508 = buf498; del buf498  # reuse
        buf509 = buf497; del buf497  # reuse
        # Topologically Sorted Source Nodes: [x_584], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf505, buf506, buf507, buf508, buf509, 1568, 3, grid=grid(1568), stream=stream0)
        del buf505
        del buf506
        del buf507
        buf512 = empty_strided_cuda((8, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_584, x_585], Original ATen: [aten.native_layer_norm, aten.mean]
        triton_per_fused_mean_native_layer_norm_13.run(buf504, buf508, buf509, arg291_1, arg292_1, buf512, 3072, 196, grid=grid(3072), stream=stream0)
        del arg291_1
        del arg292_1
        del buf504
        del buf508
        del buf509
        buf513 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_584, x_585, x_587], Original ATen: [aten.native_layer_norm, aten.mean, aten.addmm]
        extern_kernels.addmm(arg294_1, buf512, reinterpret_tensor(arg293_1, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf513)
        del arg293_1
        del arg294_1
        del buf512
    return (buf513, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gmixer_24_224', benchmark_compiled_module)
